NO_ATTEMPT=2**0
IMAGE_FLAGS=2**1
PSF_FAILURE=2**2
OBJ_FAILURE=2**3

NAME_MAP={
    'ok':0,
    0:'ok',
    'no_attempt':NO_ATTEMPT,
    NO_ATTEMPT:'no_attempt',

    'image_flags':IMAGE_FLAGS,
    IMAGE_FLAGS:'image_flags',

    'psf_failure': PSF_FAILURE,
    PSF_FAILURE:'psf_failure',
    'obj_failure': OBJ_FAILURE,
    OBJ_FAILURE:'obj_failure',
}

def get_name(val):
    return NAME_MAP[val]
