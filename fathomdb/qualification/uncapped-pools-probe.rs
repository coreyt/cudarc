//! FathomDB fork probe (R-FC-CODE-2 findings L1, L2): GPU virtual address
//! space per uncapped private pool on Tegra, and the allocator that
//! `CudaContext::new` selects once uncapped pools hold all of it.
//!
//! Built against the fork by path (`Cargo.toml` beside this file):
//! `cudarc = { package = "cudarc-fathomdb", path = "<fork tree>",
//! default-features = false, features = ["std", "driver",
//! "dynamic-loading", "cuda-12060"] }`.
//!
//! Mode `exhaust` (fresh process): create uncapped pools (`max_size` 0)
//! until `cuMemPoolCreate` fails, printing `VmSize`/`VmRSS` from
//! `/proc/self/status` before and after each create; then call
//! `cuDeviceGetDefaultMemPool` directly; then `CudaContext::new(0)` and
//! print its `alloc_mode()`. Mode `fresh`: `CudaContext::new(0)` first
//! (control), then the same pool loop.
use cudarc::driver::{result, CudaContext, CudaMemPool, MemPoolProps};

fn status(key: &str) -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = s.lines().find(|l| l.starts_with(key)).unwrap();
    let kb: u64 = line.split_whitespace().nth(1).unwrap().parse().unwrap();
    kb * 1024
}

fn vm() -> (u64, u64) {
    (status("VmSize:"), status("VmRSS:"))
}

fn exhaust() -> Vec<CudaMemPool> {
    let mut pools = Vec::new();
    loop {
        let (size_before, rss_before) = vm();
        let created = CudaMemPool::create(0, &MemPoolProps::default());
        let (size_after, rss_after) = vm();
        let n = pools.len() + 1;
        let delta = size_after as i64 - size_before as i64;
        match created {
            Ok(pool) => {
                println!(
                    "pool {n}: ok; VmSize {size_before} -> {size_after} (+{delta} bytes, {:.2} GiB); VmRSS {rss_before} -> {rss_after}",
                    delta as f64 / (1u64 << 30) as f64
                );
                pools.push(pool);
            }
            Err(e) => {
                println!("pool {n}: {e:?}; VmSize {size_before} -> {size_after} (+{delta} bytes)");
                return pools;
            }
        }
        if n >= 64 {
            println!("stopped after 64 pools without an error");
            return pools;
        }
    }
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    let (pre_size, pre_rss) = vm();
    result::init().unwrap();
    println!("before cuInit: VmSize {pre_size} VmRSS {pre_rss}");
    let dev = result::device::get(0).unwrap();
    let integrated = unsafe {
        result::device::get_attribute(
            dev,
            cudarc::driver::sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_INTEGRATED,
        )
    }
    .unwrap();
    println!(
        "mode {mode}; driver {}; integrated {integrated}; start VmSize {} VmRSS {}",
        {
            let mut v = 0;
            unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut v) };
            v
        },
        vm().0,
        vm().1
    );
    match mode.as_str() {
        "exhaust" => {
            let pools = exhaust();
            let (s0, _) = vm();
            let default_pool = unsafe { result::device::get_default_mem_pool(dev) };
            let (s1, _) = vm();
            println!(
                "with {} uncapped pools live: cuDeviceGetDefaultMemPool -> {:?}; VmSize {s0} -> {s1}",
                pools.len(),
                default_pool.map(|_| "ok")
            );
            let ctx = CudaContext::new(0).unwrap();
            println!(
                "with {} uncapped pools live: CudaContext::new(0).alloc_mode() = {:?}",
                pools.len(),
                ctx.alloc_mode()
            );
            drop(ctx);
            drop(pools);
            let ctx = CudaContext::new(0).unwrap();
            println!(
                "after dropping the pools: CudaContext::new(0).alloc_mode() = {:?} (process-wide decision, fixed by the first new)",
                ctx.alloc_mode()
            );
        }
        "fresh" => {
            let (s0, _) = vm();
            let ctx = CudaContext::new(0).unwrap();
            let (s1, _) = vm();
            println!(
                "fresh process: CudaContext::new(0).alloc_mode() = {:?}; VmSize {s0} -> {s1} (+{} bytes)",
                ctx.alloc_mode(),
                s1 as i64 - s0 as i64
            );
            let pools = exhaust();
            println!("with the context's default pool: {} uncapped private pools created", pools.len());
        }
        _ => panic!("usage: uncapped-pools-probe exhaust|fresh"),
    }
}
