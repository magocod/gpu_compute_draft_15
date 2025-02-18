use io::Error as IoError;
use opencl::error::OclError;
use opencl::opencl_sys::bindings::cl_int;
use std::io;

// opencl collection error codes

// stack
pub const CL_COLLECTION_INVALID_STACK_ID: cl_int = -200;

// set
pub const CL_COLLECTION_INVALID_ARRAY_SET_ID: cl_int = -300;

// dict
pub const CL_COLLECTION_INVALID_DICT_ID: cl_int = -400;

// queue
pub const CL_COLLECTION_INVALID_QUEUE_ID: cl_int = -500;

// cache
pub const CL_COLLECTION_INVALID_MINI_LRU_ID: cl_int = -600;
pub const CL_COLLECTION_INVALID_LRU_ID: cl_int = -700;

// map
pub const CL_COLLECTION_INVALID_MAP_VALUE_LEN: cl_int = -800;

#[derive(Debug, PartialEq)]
pub enum OpenclError {
    // original opencl error code
    OpenCl(cl_int),
    // opencl wrapper error code
    OpenClWrapper(cl_int),
    // opencl collection error code
    OpenclCollection(cl_int),
    Unknown(&'static str),
}

pub type OpenClResult<T> = Result<T, OpenclError>;

impl std::error::Error for OpenclError {}

impl std::fmt::Display for OpenclError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenclError::OpenCl(v) => write!(f, "opencl error code: {v}"),
            OpenclError::OpenClWrapper(v) => write!(f, "opencl wrapper error code: {v}"),
            OpenclError::OpenclCollection(v) => write!(f, "opencl collection error code: {v}"),
            OpenclError::Unknown(message) => write!(f, "{message}"),
        }
    }
}

impl From<OclError> for OpenclError {
    fn from(e: OclError) -> Self {
        match e {
            OclError::Code(code) => OpenclError::OpenCl(code),
            OclError::Wrapper(code) => OpenclError::OpenClWrapper(code),
        }
    }
}

impl From<OpenclError> for IoError {
    fn from(e: OpenclError) -> Self {
        let error_kind = io::ErrorKind::Other;

        match e {
            OpenclError::OpenCl(code) => {
                Self::new(error_kind, format!("opencl error code: {code}"))
            }
            OpenclError::OpenClWrapper(code) => {
                Self::new(error_kind, format!("opencl wrapper error code: {code}"))
            }
            OpenclError::OpenclCollection(code) => Self::new(
                error_kind,
                format!("opencl collection error code: {}", code),
            ),
            OpenclError::Unknown(message) => {
                Self::new(error_kind, format!("cl unknown error: {}", message))
            }
        }
    }
}

// TODO complete handle IoError
impl From<IoError> for OpenclError {
    fn from(_e: IoError) -> Self {
        // match e.kind() {
        //     ErrorKind::NotFound => {}
        //     ErrorKind::PermissionDenied => {}
        //     ErrorKind::ConnectionRefused => {}
        //     ErrorKind::ConnectionReset => {}
        //     ErrorKind::ConnectionAborted => {}
        //     ErrorKind::NotConnected => {}
        //     ErrorKind::AddrInUse => {}
        //     ErrorKind::AddrNotAvailable => {}
        //     ErrorKind::BrokenPipe => {}
        //     ErrorKind::AlreadyExists => {}
        //     ErrorKind::WouldBlock => {}
        //     ErrorKind::InvalidInput => {}
        //     ErrorKind::InvalidData => {}
        //     ErrorKind::TimedOut => {}
        //     ErrorKind::WriteZero => {}
        //     ErrorKind::Interrupted => {}
        //     ErrorKind::Unsupported => {}
        //     ErrorKind::UnexpectedEof => {}
        //     ErrorKind::OutOfMemory => {}
        //     ErrorKind::Other => {}
        //     _ => {}
        // }
        OpenclError::Unknown("io error")
    }
}
