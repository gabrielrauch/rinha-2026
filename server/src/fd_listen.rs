//! FD-passing receiver: pair to the SCM_RIGHTS-sending LB. When the API is
//! launched with `RINHA_FD_SOCK=<path>`, we open a UDS at that path and
//! receive accepted TCP FDs from the LB (1 byte payload + ancillary FD per
//! message). Each FD becomes a `monoio::net::TcpStream` we can serve HTTP on
//! directly — no byte proxy between LB and us.
//!
//! Architecture:
//!   - OS thread A: `accept()` on the UDS listener, spawning per-control-conn
//!     recv threads as LB workers connect.
//!   - OS thread per control-conn: `recvmsg` loop that pulls FDs out and
//!     pushes them onto a channel.
//!   - monoio runtime: wakes on a Unix socketpair byte whenever the channel
//!     has new work; for each FD, wraps it as a `TcpStream` and spawns the
//!     normal `wire::handle_connection`.

use anyhow::Result;
use std::os::unix::io::{FromRawFd, RawFd};
use std::path::Path;
use std::sync::mpsc;
use std::sync::Arc;

use crate::blob::Blob;
use monoio::io::AsyncReadRent;
use monoio::net::{TcpStream, UnixStream};

type Driver = monoio::FusionDriver;

pub fn run(blob: Arc<Blob>, sock_path: &Path) -> Result<()> {
    if let Some(parent) = sock_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let _ = std::fs::remove_file(sock_path);

    let listener_fd = create_unix_listener(sock_path)?;
    eprintln!("listening on uds fd-passing {}", sock_path.display());

    let (fd_tx, fd_rx) = mpsc::channel::<RawFd>();
    let (wake_writer_fd, wake_reader_fd) = socketpair_nonblocking()?;

    // Accept thread: per-control-conn recv threads
    std::thread::Builder::new()
        .name("fd-accept".into())
        .spawn(move || {
            accept_loop(listener_fd, fd_tx, wake_writer_fd);
        })?;

    // monoio runtime drains the channel and spawns per-FD handlers.
    let mut rt = monoio::RuntimeBuilder::<Driver>::new()
        .enable_timer()
        .build()
        .expect("monoio runtime build");
    rt.block_on(async move {
        let std_wake = unsafe { std::os::unix::net::UnixStream::from_raw_fd(wake_reader_fd) };
        std_wake.set_nonblocking(false).ok();
        let mut wake = UnixStream::from_std(std_wake)?;
        let mut buf = vec![0u8; 256];
        loop {
            // Block until LB sends us at least one FD.
            let (res, b) = wake.read(buf).await;
            buf = b;
            match res {
                Ok(0) => break,
                Ok(_) => {
                    while let Ok(fd) = fd_rx.try_recv() {
                        let std_tcp = unsafe { std::net::TcpStream::from_raw_fd(fd) };
                        if let Err(e) = std_tcp.set_nonblocking(true) {
                            eprintln!("set_nonblocking failed: {e}");
                            continue;
                        }
                        let stream = match TcpStream::from_std(std_tcp) {
                            Ok(s) => s,
                            Err(e) => {
                                eprintln!("TcpStream::from_std failed: {e}");
                                continue;
                            }
                        };
                        let blob = blob.clone();
                        monoio::spawn(async move {
                            crate::wire::handle_connection(blob, stream).await;
                        });
                    }
                }
                Err(e) => {
                    eprintln!("wake socket read failed: {e}");
                    break;
                }
            }
            // Resize buf back to read capacity since `read` may have shrunk it.
            if buf.capacity() < 256 {
                buf = vec![0u8; 256];
            } else {
                unsafe { buf.set_len(buf.capacity()) };
            }
        }
        Ok::<_, anyhow::Error>(())
    })
}

fn create_unix_listener(path: &Path) -> Result<libc::c_int> {
    let fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let bytes = path.as_os_str().as_encoded_bytes();
    if bytes.len() >= addr.sun_path.len() {
        unsafe { libc::close(fd) };
        anyhow::bail!("uds path too long");
    }
    for (i, &b) in bytes.iter().enumerate() {
        addr.sun_path[i] = b as libc::c_char;
    }
    let r = unsafe {
        libc::bind(
            fd,
            &addr as *const _ as *const libc::sockaddr,
            std::mem::size_of::<libc::sockaddr_un>() as libc::socklen_t,
        )
    };
    if r != 0 {
        let e = std::io::Error::last_os_error();
        unsafe { libc::close(fd) };
        return Err(e.into());
    }
    // World-RW so the LB can connect from any uid the container assigns.
    unsafe {
        let cpath = std::ffi::CString::new(bytes)?;
        libc::chmod(cpath.as_ptr(), 0o666);
    }
    let r = unsafe { libc::listen(fd, 1024) };
    if r != 0 {
        let e = std::io::Error::last_os_error();
        unsafe { libc::close(fd) };
        return Err(e.into());
    }
    Ok(fd)
}

fn accept_loop(listener_fd: libc::c_int, fd_tx: mpsc::Sender<RawFd>, wake_writer_fd: RawFd) {
    loop {
        let control = unsafe {
            libc::accept4(
                listener_fd,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                libc::SOCK_CLOEXEC,
            )
        };
        if control < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            eprintln!("fd-accept error: {err}");
            continue;
        }
        let fd_tx = fd_tx.clone();
        let wake = wake_writer_fd;
        std::thread::Builder::new()
            .name("fd-recv".into())
            .spawn(move || recv_loop(control, fd_tx, wake))
            .expect("spawn fd-recv");
    }
}

fn recv_loop(control_fd: libc::c_int, fd_tx: mpsc::Sender<RawFd>, wake_writer_fd: RawFd) {
    while let Some(fd) = recv_fd(control_fd) {
        if fd_tx.send(fd).is_err() {
            unsafe { libc::close(fd) };
            break;
        }
        let one: u8 = 1;
        unsafe {
            libc::write(wake_writer_fd, &one as *const _ as *const _, 1);
        }
    }
    unsafe { libc::close(control_fd) };
}

fn recv_fd(control_fd: libc::c_int) -> Option<RawFd> {
    let mut one: u8 = 0;
    let mut iov = libc::iovec {
        iov_base: &mut one as *mut _ as *mut _,
        iov_len: 1,
    };
    let mut cmsg_buf = [0u8; 64];
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.as_mut_ptr() as *mut _;
    msg.msg_controllen = cmsg_buf.len();
    loop {
        let n = unsafe { libc::recvmsg(control_fd, &mut msg, 0) };
        if n < 0 {
            let e = std::io::Error::last_os_error();
            if e.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return None;
        }
        if n == 0 {
            return None;
        }
        unsafe {
            let mut cmsg = libc::CMSG_FIRSTHDR(&msg);
            while !cmsg.is_null() {
                if (*cmsg).cmsg_level == libc::SOL_SOCKET
                    && (*cmsg).cmsg_type == libc::SCM_RIGHTS
                    && (*cmsg).cmsg_len
                        >= libc::CMSG_LEN(std::mem::size_of::<libc::c_int>() as u32) as _
                {
                    let mut fd: libc::c_int = -1;
                    std::ptr::copy_nonoverlapping(
                        libc::CMSG_DATA(cmsg) as *const u8,
                        &mut fd as *mut libc::c_int as *mut u8,
                        std::mem::size_of::<libc::c_int>(),
                    );
                    return Some(fd);
                }
                cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
            }
            return None;
        }
    }
}

fn socketpair_nonblocking() -> Result<(RawFd, RawFd)> {
    let mut fds: [libc::c_int; 2] = [0; 2];
    let r = unsafe {
        libc::socketpair(
            libc::AF_UNIX,
            libc::SOCK_STREAM | libc::SOCK_CLOEXEC,
            0,
            fds.as_mut_ptr(),
        )
    };
    if r != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok((fds[0], fds[1]))
}
