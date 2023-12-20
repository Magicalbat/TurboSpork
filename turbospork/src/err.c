#include "err.h"

#include <stdio.h>

static const ts_string8 _error_strings[TS_ERR_COUNT] = {
#define X(code) { (ts_u64)sizeof(#code), (ts_u8*)#code },
    TS_ERROR_XLIST
#undef X
};

static void _default_error_callback(ts_error err) {
    ts_string8 code_str = ts_err_to_str(err.code);

    fprintf(stderr, "TurboSpork %.*s: \"%.*s\"\n", (int)code_str.size, (char*)code_str.str, (int)err.msg.size, (char*)err.msg.str);
}

static ts_error_callback* _error_callback = _default_error_callback;

void ts_err(ts_error err) {
    _error_callback(err);
}

void ts_err_set_callback(ts_error_callback* callback) {
    _error_callback = callback;
}

ts_string8 ts_err_to_str(ts_error_code code) {
    if (code >= TS_ERR_COUNT) {
        return _error_strings[0];
    }

    return _error_strings[code];
}
ts_error_code ts_err_from_str(ts_string8 str) {
    ts_error_code out = 0;

    for (ts_u32 i = 0; i < TS_ERR_COUNT; i++) {
        if (ts_str8_equals(str, _error_strings[i])) {
            out = i;
            break;
        }
    }

    return out;
}

