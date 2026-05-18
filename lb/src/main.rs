//! Multi-worker LB that hands accepted TCP sockets over to the API workers
//! via SCM_RIGHTS on a Unix-domain control channel. Mirrors MXLange's `lb.c`
//! so the LB never touches request bytes — it just opens the door, the API
//! does the rest.
//!
//! For each accepted TCP connection:
//!   1. Pick an upstream by round-robin (atomic counter).
//!   2. Open or reuse this worker's persistent UDS to that upstream.
//!   3. `sendmsg(SCM_RIGHTS)` the FD across with a single byte payload.
//!   4. Close our local TCP FD; the kernel keeps the underlying socket
//!      alive in the API process.
//!
//! Architecture: one OS thread per worker (default 2). The kernel
//! distributes `accept` across them via `SO_REUSEPORT`. Each worker owns
//! its own Vec<Sender> so the persistent UDS FDs are touched by exactly
//! one thread.

#[cfg(not(target_os = "linux"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("this LB only runs on Linux (relies on SCM_RIGHTS + accept4)")
}

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    use std::os::unix::io::RawFd;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .or_else(|| {
            std::env::var("BIND_ADDR")
                .ok()
                .and_then(|s| s.rsplit(':').next().and_then(|p| p.parse().ok()))
        })
        .unwrap_or(9999);
    let upstreams_raw = std::env::var("FD_UPSTREAMS")
        .or_else(|_| std::env::var("UPSTREAMS"))
        .unwrap_or_else(|_| "/run/sock/api1.sock,/run/sock/api2.sock".into());
    let upstreams: Vec<String> = upstreams_raw
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    anyhow::ensure!(!upstreams.is_empty(), "no upstreams configured");

    let workers: usize = std::env::var("WORKERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    // Wait for upstreams to come up — they may still be opening their UDS
    // listener at LB start. We don't want to crash if the LB races the API.
    eprintln!(
        "lb starting port={} workers={} upstreams={}",
        port,
        workers,
        upstreams.len()
    );
    wait_upstreams(&upstreams);
    eprintln!("lb upstreams ready");

    let listener_fd = create_tcp_listener(port)
        .map_err(|e| anyhow::anyhow!("create TCP listener on :{port}: {e}"))?;
    let upstreams = Arc::new(upstreams);
    let next = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let upstreams = upstreams.clone();
        let next = next.clone();
        let fd: RawFd = listener_fd;
        handles.push(std::thread::spawn(move || {
            worker_loop(fd, &upstreams, &next);
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn create_tcp_listener(port: u16) -> std::io::Result<std::os::unix::io::RawFd> {
    use std::os::unix::io::IntoRawFd;
    let sock = socket_set_reuse(libc::AF_INET, libc::SOCK_STREAM | libc::SOCK_CLOEXEC)?;
    let one: libc::c_int = 1;
    unsafe {
        libc::setsockopt(
            sock,
            libc::SOL_SOCKET,
            libc::SO_REUSEPORT,
            &one as *const _ as *const _,
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        );
    }
    let mut addr: libc::sockaddr_in = unsafe { std::mem::zeroed() };
    addr.sin_family = libc::AF_INET as libc::sa_family_t;
    addr.sin_addr.s_addr = u32::to_be(libc::INADDR_ANY);
    addr.sin_port = u16::to_be(port);
    let r = unsafe {
        libc::bind(
            sock,
            &addr as *const _ as *const libc::sockaddr,
            std::mem::size_of::<libc::sockaddr_in>() as libc::socklen_t,
        )
    };
    if r != 0 {
        let e = std::io::Error::last_os_error();
        unsafe { libc::close(sock) };
        return Err(e);
    }
    let r = unsafe { libc::listen(sock, 8192) };
    if r != 0 {
        let e = std::io::Error::last_os_error();
        unsafe { libc::close(sock) };
        return Err(e);
    }
    // We need a stable RawFd value — use IntoRawFd via a TcpListener wrapper to leak it.
    let listener = unsafe { std::net::TcpListener::from_raw_fd(sock) };
    Ok(listener.into_raw_fd())
}

#[cfg(target_os = "linux")]
fn socket_set_reuse(family: libc::c_int, kind: libc::c_int) -> std::io::Result<libc::c_int> {
    let sock = unsafe { libc::socket(family, kind, 0) };
    if sock < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let one: libc::c_int = 1;
    unsafe {
        libc::setsockopt(
            sock,
            libc::SOL_SOCKET,
            libc::SO_REUSEADDR,
            &one as *const _ as *const _,
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        );
    }
    Ok(sock)
}

#[cfg(target_os = "linux")]
fn connect_unix(path: &str) -> std::io::Result<libc::c_int> {
    let fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let path_bytes = path.as_bytes();
    if path_bytes.len() >= addr.sun_path.len() {
        unsafe { libc::close(fd) };
        return Err(std::io::Error::other("uds path too long"));
    }
    for (i, &b) in path_bytes.iter().enumerate() {
        addr.sun_path[i] = b as libc::c_char;
    }
    let r = unsafe {
        libc::connect(
            fd,
            &addr as *const _ as *const libc::sockaddr,
            std::mem::size_of::<libc::sockaddr_un>() as libc::socklen_t,
        )
    };
    if r != 0 {
        let e = std::io::Error::last_os_error();
        unsafe { libc::close(fd) };
        return Err(e);
    }
    Ok(fd)
}

#[cfg(target_os = "linux")]
fn wait_upstreams(upstreams: &[String]) {
    let delay = std::time::Duration::from_millis(50);
    for _ in 0..200 {
        let ready = upstreams
            .iter()
            .filter(|p| match connect_unix(p) {
                Ok(fd) => {
                    unsafe { libc::close(fd) };
                    true
                }
                Err(_) => false,
            })
            .count();
        if ready == upstreams.len() {
            return;
        }
        std::thread::sleep(delay);
    }
    eprintln!("lb: upstreams not all up after 10s, starting anyway");
}

#[cfg(target_os = "linux")]
fn worker_loop(
    listener_fd: libc::c_int,
    upstreams: &[String],
    next: &std::sync::atomic::AtomicUsize,
) {
    use std::sync::atomic::Ordering;
    // Each worker keeps its own persistent UDS FD per upstream.
    let mut senders: Vec<Sender> = upstreams.iter().map(|p| Sender::new(p.clone())).collect();
    loop {
        let client = unsafe {
            libc::accept4(
                listener_fd,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                libc::SOCK_CLOEXEC,
            )
        };
        if client < 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            eprintln!("lb accept error: {err}");
            continue;
        }
        set_tcp_nodelay(client);

        let start = next.fetch_add(1, Ordering::Relaxed);
        let mut sent = false;
        for i in 0..senders.len() {
            let idx = (start + i) % senders.len();
            if senders[idx].send(client) {
                sent = true;
                break;
            }
        }
        if !sent {
            write_bad_gateway(client);
        }
        unsafe { libc::close(client) };
    }
}

#[cfg(target_os = "linux")]
fn set_tcp_nodelay(fd: libc::c_int) {
    let one: libc::c_int = 1;
    unsafe {
        libc::setsockopt(
            fd,
            libc::IPPROTO_TCP,
            libc::TCP_NODELAY,
            &one as *const _ as *const _,
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        );
    }
}

#[cfg(target_os = "linux")]
struct Sender {
    path: String,
    fd: libc::c_int,
}

#[cfg(target_os = "linux")]
impl Sender {
    fn new(path: String) -> Self {
        Self { path, fd: -1 }
    }

    /// Try to ship `client_fd`. Reconnect once on failure.
    fn send(&mut self, client_fd: libc::c_int) -> bool {
        if self.fd < 0 {
            self.fd = connect_unix(&self.path).unwrap_or(-1);
            if self.fd < 0 {
                return false;
            }
        }
        if send_fd_once(self.fd, client_fd) {
            return true;
        }
        unsafe { libc::close(self.fd) };
        self.fd = connect_unix(&self.path).unwrap_or(-1);
        if self.fd < 0 {
            return false;
        }
        send_fd_once(self.fd, client_fd)
    }
}

#[cfg(target_os = "linux")]
impl Drop for Sender {
    fn drop(&mut self) {
        if self.fd >= 0 {
            unsafe { libc::close(self.fd) };
        }
    }
}

#[cfg(target_os = "linux")]
fn send_fd_once(control_fd: libc::c_int, client_fd: libc::c_int) -> bool {
    let one: u8 = 0;
    let mut iov = libc::iovec {
        iov_base: &one as *const _ as *mut _,
        iov_len: 1,
    };
    // CMSG_SPACE(sizeof(int)) - the buffer must be big enough for the header
    // plus the FD payload plus alignment slack.
    let mut cmsg_buf = [0u8; 24]; // CMSG_SPACE(4) = 24 on Linux 64-bit
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.as_mut_ptr() as *mut _;
    msg.msg_controllen = cmsg_buf.len();

    unsafe {
        let cmsg = libc::CMSG_FIRSTHDR(&msg);
        if cmsg.is_null() {
            return false;
        }
        (*cmsg).cmsg_level = libc::SOL_SOCKET;
        (*cmsg).cmsg_type = libc::SCM_RIGHTS;
        (*cmsg).cmsg_len = libc::CMSG_LEN(std::mem::size_of::<libc::c_int>() as u32) as _;
        std::ptr::copy_nonoverlapping(
            &client_fd as *const libc::c_int as *const u8,
            libc::CMSG_DATA(cmsg),
            std::mem::size_of::<libc::c_int>(),
        );
        msg.msg_controllen = (*cmsg).cmsg_len;

        loop {
            let n = libc::sendmsg(control_fd, &msg, libc::MSG_NOSIGNAL);
            if n == 1 {
                return true;
            }
            if n < 0 {
                let e = std::io::Error::last_os_error();
                if e.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
            }
            return false;
        }
    }
}

#[cfg(target_os = "linux")]
fn write_bad_gateway(fd: libc::c_int) {
    const RESP: &[u8] =
        b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    let mut p = RESP.as_ptr();
    let mut n = RESP.len();
    while n > 0 {
        let w = unsafe { libc::write(fd, p as *const _, n) };
        if w < 0 {
            let e = std::io::Error::last_os_error();
            if e.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return;
        }
        if w == 0 {
            return;
        }
        p = unsafe { p.add(w as usize) };
        n -= w as usize;
    }
}

#[cfg(target_os = "linux")]
use std::os::unix::io::FromRawFd;
