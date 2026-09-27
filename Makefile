ID := sen-tts
DEV_PORT := 4964
CARGO_BUILD_JOBS ?= 4
# Shared with sen-sysone and sen-ocr while developing on the same machine:
# common deps (tokio, axum, ort, tokenizers, …) compile once instead of three
# times. Override for CI, where each platform job gets its own runner.
CARGO_TARGET_DIR ?= ../.cargo-target-cpu
CARGO_INCREMENTAL ?= 0
export CARGO_BUILD_JOBS CARGO_TARGET_DIR CARGO_INCREMENTAL

VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    PLATFORM := darwin-arm64
  else
    PLATFORM := darwin-x64
  endif
else ifeq ($(UNAME_S),Linux)
  ifeq ($(UNAME_M),aarch64)
    PLATFORM := linux-arm64
  else
    PLATFORM := linux-x64
  endif
else
  PLATFORM := windows-x64
endif

DIST := dist
PKG_NAME := $(ID)-$(VERSION)-$(PLATFORM)
PKG_DIR := $(DIST)/$(PKG_NAME)

.PHONY: build test package install-local run-dev clean

build:
	cargo build --release

test:
	cargo test

# dist/<id>-<version>-<platform>.tar.gz (+ .sha256), manifest at the archive's
# top level beside bin/ — see senclaw/docs/runtime-protocol.md §2.3.
package: build
	rm -rf "$(PKG_DIR)"
	mkdir -p "$(PKG_DIR)/bin"
	cp "$(CARGO_TARGET_DIR)/release/$(ID)" "$(PKG_DIR)/bin/$(ID)"
	sh scripts/bundle-shared-libs.sh "$(PKG_DIR)/bin/$(ID)"
	cp senclaw-runtime.json "$(PKG_DIR)/senclaw-runtime.json"
	cd "$(DIST)" && tar czf "$(PKG_NAME).tar.gz" -C "$(PKG_NAME)" .
	cd "$(DIST)" && shasum -a 256 "$(PKG_NAME).tar.gz" > "$(PKG_NAME).tar.gz.sha256"
	@echo "packaged $(DIST)/$(PKG_NAME).tar.gz"

# `senclaw runtime install-local <archive>` when a `senclaw` binary is on
# PATH; otherwise extract straight into ~/.senclaw/runtimes/<id>/<version>/,
# writing the manifest last (its absence marks an interrupted install).
install-local: package
	@if command -v senclaw >/dev/null 2>&1; then \
		senclaw runtime install-local "$(DIST)/$(PKG_NAME).tar.gz"; \
	else \
		dst="$$HOME/.senclaw/runtimes/$(ID)/$(VERSION)"; \
		echo "no senclaw binary on PATH; extracting into $$dst"; \
		rm -rf "$$dst"; mkdir -p "$$dst/bin"; \
		cp "$(PKG_DIR)/bin/$(ID)" "$$dst/bin/$(ID)"; \
		for f in "$(PKG_DIR)"/bin/*.dylib "$(PKG_DIR)"/bin/*.so; do \
			[ -e "$$f" ] || continue; cp "$$f" "$$dst/bin/$$(basename "$$f")" 2>/dev/null || true; \
		done; \
		cp "$(PKG_DIR)/senclaw-runtime.json" "$$dst/senclaw-runtime.json"; \
		echo "installed into $$dst"; \
	fi

# Standalone serve on a fixed dev port: no token (no auth), no parent (no
# watchdog) — LaunchEnv::from_env's standalone defaults.
run-dev:
	env -u SENCLAW_RUNTIME_TOKEN -u SENCLAW_PARENT_PID cargo run --release -- serve --host 127.0.0.1 --port $(DEV_PORT)

clean:
	cargo clean -p $(ID)
	rm -rf $(DIST)
