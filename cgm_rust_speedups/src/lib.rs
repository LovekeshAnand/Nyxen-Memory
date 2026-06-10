use std::ptr;

// Structure matching NVML memory struct
#[repr(C)]
pub struct NvmlMemory {
    pub total: u64,
    pub free: u64,
    pub used: u64,
}

// Windows winapi equivalents for memory query
#[cfg(target_os = "windows")]
mod win {
    #[repr(C)]
    pub struct MemoryStatusEx {
        pub dw_length: u32,
        pub dw_memory_load: u32,
        pub ull_total_phys: u64,
        pub ull_avail_phys: u64,
        pub ull_total_page_file: u64,
        pub ull_avail_page_file: u64,
        pub ull_total_virtual: u64,
        pub ull_avail_virtual: u64,
        pub ull_avail_extended_virtual: u64,
    }

    extern "system" {
        pub fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
    }
}

// 1. System RAM check
#[no_mangle]
pub unsafe extern "C" fn get_system_ram(free_bytes: *mut u64, total_bytes: *mut u64) -> i32 {
    if free_bytes.is_null() || total_bytes.is_null() {
        return -1;
    }
    #[cfg(target_os = "windows")]
    {
        let mut mem_info: win::MemoryStatusEx = std::mem::zeroed();
        mem_info.dw_length = std::mem::size_of::<win::MemoryStatusEx>() as u32;
        if win::GlobalMemoryStatusEx(&mut mem_info) != 0 {
            *free_bytes = mem_info.ull_avail_phys;
            *total_bytes = mem_info.ull_total_phys;
            return 0;
        }
        return -2;
    }
    #[cfg(not(target_os = "windows"))]
    {
        return -3; // Unsupported OS
    }
}

// 2. GPU VRAM check using dynamic NVML loading
#[no_mangle]
pub unsafe extern "C" fn get_gpu_vram(free_bytes: *mut u64, total_bytes: *mut u64) -> i32 {
    if free_bytes.is_null() || total_bytes.is_null() {
        return -1;
    }
    #[cfg(target_os = "windows")]
    {
        use std::ffi::CString;
        extern "system" {
            fn LoadLibraryA(lp_lib_file_name: *const i8) -> *mut std::ffi::c_void;
            fn GetProcAddress(h_module: *mut std::ffi::c_void, lp_proc_name: *const i8) -> *mut std::ffi::c_void;
            fn FreeLibrary(h_module: *mut std::ffi::c_void) -> i32;
        }

        let nvml_names = vec!["nvml.dll", "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll", "C:\\Windows\\System32\\nvml.dll"];
        let mut nvml_lib = ptr::null_mut();
        for name in nvml_names {
            let c_name = CString::new(name).unwrap();
            nvml_lib = LoadLibraryA(c_name.as_ptr());
            if !nvml_lib.is_null() {
                break;
            }
        }
        if nvml_lib.is_null() {
            return -2;
        }

        let fn_init = GetProcAddress(nvml_lib, CString::new("nvmlInit").unwrap().as_ptr());
        let fn_shutdown = GetProcAddress(nvml_lib, CString::new("nvmlShutdown").unwrap().as_ptr());
        let fn_device_get_handle = GetProcAddress(nvml_lib, CString::new("nvmlDeviceGetHandleByIndex").unwrap().as_ptr());
        let fn_device_get_mem = GetProcAddress(nvml_lib, CString::new("nvmlDeviceGetMemoryInfo").unwrap().as_ptr());

        if fn_init.is_null() || fn_shutdown.is_null() || fn_device_get_handle.is_null() || fn_device_get_mem.is_null() {
            FreeLibrary(nvml_lib);
            return -3;
        }

        let nvml_init: unsafe extern "C" fn() -> i32 = std::mem::transmute(fn_init);
        let nvml_shutdown: unsafe extern "C" fn() -> i32 = std::mem::transmute(fn_shutdown);
        let nvml_device_get_handle_by_index: unsafe extern "C" fn(u32, *mut *mut std::ffi::c_void) -> i32 = std::mem::transmute(fn_device_get_handle);
        let nvml_device_get_memory_info: unsafe extern "C" fn(*mut std::ffi::c_void, *mut NvmlMemory) -> i32 = std::mem::transmute(fn_device_get_mem);

        if nvml_init() != 0 {
            FreeLibrary(nvml_lib);
            return -4;
        }

        let mut device_handle: *mut std::ffi::c_void = ptr::null_mut();
        if nvml_device_get_handle_by_index(0, &mut device_handle) != 0 {
            nvml_shutdown();
            FreeLibrary(nvml_lib);
            return -5;
        }

        let mut mem_info: NvmlMemory = std::mem::zeroed();
        if nvml_device_get_memory_info(device_handle, &mut mem_info) != 0 {
            nvml_shutdown();
            FreeLibrary(nvml_lib);
            return -6;
        }

        *free_bytes = mem_info.free;
        *total_bytes = mem_info.total;

        nvml_shutdown();
        FreeLibrary(nvml_lib);
        0
    }
    #[cfg(not(target_os = "windows"))]
    {
        return -7;
    }
}

// 3. Fast cosine similarity
#[no_mangle]
pub extern "C" fn compute_cosine_similarity(a: *const f32, b: *const f32, dim: i32) -> f32 {
    if a.is_null() || b.is_null() || dim <= 0 {
        return 0.0;
    }
    unsafe {
        let slice_a = std::slice::from_raw_parts(a, dim as usize);
        let slice_b = std::slice::from_raw_parts(b, dim as usize);
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        for i in 0..(dim as usize) {
            dot += slice_a[i] * slice_b[i];
            norm_a += slice_a[i] * slice_a[i];
            norm_b += slice_b[i] * slice_b[i];
        }
        if norm_a <= 0.0 || norm_b <= 0.0 {
            return 0.0;
        }
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

// 4. Dynamic GPU Temperature Query
#[no_mangle]
pub unsafe extern "C" fn get_gpu_temperature(temp: *mut u32) -> i32 {
    if temp.is_null() {
        return -1;
    }
    #[cfg(target_os = "windows")]
    {
        use std::ffi::CString;
        extern "system" {
            fn LoadLibraryA(lp_lib_file_name: *const i8) -> *mut std::ffi::c_void;
            fn GetProcAddress(h_module: *mut std::ffi::c_void, lp_proc_name: *const i8) -> *mut std::ffi::c_void;
            fn FreeLibrary(h_module: *mut std::ffi::c_void) -> i32;
        }

        let nvml_names = vec![
            "nvml.dll",
            "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll",
            "C:\\Windows\\System32\\nvml.dll",
        ];
        let mut nvml_lib = ptr::null_mut();
        for name in nvml_names {
            let c_name = CString::new(name).unwrap();
            nvml_lib = LoadLibraryA(c_name.as_ptr());
            if !nvml_lib.is_null() {
                break;
            }
        }
        if nvml_lib.is_null() {
            return -2;
        }

        let fn_init = GetProcAddress(nvml_lib, CString::new("nvmlInit").unwrap().as_ptr());
        let fn_shutdown = GetProcAddress(nvml_lib, CString::new("nvmlShutdown").unwrap().as_ptr());
        let fn_device_get_handle = GetProcAddress(nvml_lib, CString::new("nvmlDeviceGetHandleByIndex").unwrap().as_ptr());
        let fn_device_get_temp = GetProcAddress(nvml_lib, CString::new("nvmlDeviceGetTemperature").unwrap().as_ptr());

        if fn_init.is_null() || fn_shutdown.is_null() || fn_device_get_handle.is_null() || fn_device_get_temp.is_null() {
            FreeLibrary(nvml_lib);
            return -3;
        }

        let nvml_init: unsafe extern "C" fn() -> i32 = std::mem::transmute(fn_init);
        let nvml_shutdown: unsafe extern "C" fn() -> i32 = std::mem::transmute(fn_shutdown);
        let nvml_device_get_handle_by_index: unsafe extern "C" fn(u32, *mut *mut std::ffi::c_void) -> i32 = std::mem::transmute(fn_device_get_handle);
        let nvml_device_get_temperature: unsafe extern "C" fn(*mut std::ffi::c_void, u32, *mut u32) -> i32 = std::mem::transmute(fn_device_get_temp);

        if nvml_init() != 0 {
            FreeLibrary(nvml_lib);
            return -4;
        }

        let mut device_handle: *mut std::ffi::c_void = ptr::null_mut();
        if nvml_device_get_handle_by_index(0, &mut device_handle) != 0 {
            nvml_shutdown();
            FreeLibrary(nvml_lib);
            return -5;
        }

        let mut current_temp: u32 = 0;
        // 0 corresponds to NVML_TEMPERATURE_GPU
        if nvml_device_get_temperature(device_handle, 0, &mut current_temp) != 0 {
            nvml_shutdown();
            FreeLibrary(nvml_lib);
            return -6;
        }

        *temp = current_temp;

        nvml_shutdown();
        FreeLibrary(nvml_lib);
        0
    }
    #[cfg(not(target_os = "windows"))]
    {
        return -7;
    }
}

