//! Panic-free console logging for daemon runtime paths.
//!
//! `eprintln!`/`println!` **panic** when the write fails — and under the
//! desktop-app supervisor the daemon's stdio are pipes that can close
//! (observed: "failed printing to stderr: Broken pipe (os error 32)" mid-
//! transcription, which poisoned the WhisperEngine mutex and killed STT until
//! restart). Any *runtime* print reachable from `senclaw start` must therefore
//! go through these macros, which discard write errors instead of panicking.
//!
//! Guidance:
//! - Runtime daemon code → [`safe_eprintln!`] / [`safe_println!`] (or `tracing`).
//! - CLI subcommand output and `#[cfg(test)]` prints → plain `println!` /
//!   `eprintln!` are fine (interactive stdio, panicking on EPIPE is acceptable).

/// `eprintln!` that ignores write failures (EPIPE-safe).
#[macro_export]
macro_rules! safe_eprintln {
    ($($arg:tt)*) => {{
        use ::std::io::Write as _;
        let _ = writeln!(::std::io::stderr(), $($arg)*);
    }};
}

/// `println!` that ignores write failures (EPIPE-safe).
#[macro_export]
macro_rules! safe_println {
    ($($arg:tt)*) => {{
        use ::std::io::Write as _;
        let _ = writeln!(::std::io::stdout(), $($arg)*);
    }};
}

#[cfg(test)]
mod tests {
    #[test]
    fn safe_macros_accept_format_args_and_do_not_panic() {
        // Smoke: normal formatting works on healthy stdio.
        safe_eprintln!("safe_log test err {} {:?}", 1, "x");
        safe_println!("safe_log test out {}", 2.5);
    }
}
