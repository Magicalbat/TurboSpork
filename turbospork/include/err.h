/**
 * @file err.h
 * @brief Error handling
 */

#ifndef ERR_H
#define ERR_H

#include "base_defs.h"
#include "str.h"

/**
 * @brief List of error codes
 *
 * To use, define a macro X, insert the definition, and then undef the macro X
 */
#define TS_ERROR_XLIST \
    X(ERR_NULL) \
    X(ERR_INVALID_INPUT) \
    X(ERR_ALLOC_SIZE) \
    X(ERR_BAD_SHAPE) \
    X(ERR_BUFFER_NOT_FILLED) \
    X(ERR_PARSE) \
    X(ERR_IO) \
    X(ERR_OS) \
    X(ERR_THREADING) \
    X(ERR_INVALID_ENUM)

/**
 * @brief Error codes
 *
 * See `TS_ERROR_XLIST` for full list
 */
typedef enum {
#define X(code) TS_##code,
    TS_ERROR_XLIST
#undef X

    TS_ERR_COUNT
} ts_error_code;

/**
 * @brief Error code and message
 *
 * Used for error callbacks
 */
typedef struct {
    ts_error_code code;
    ts_string8 msg;
} ts_error;

/// Error callback function
typedef void (ts_error_callback)(ts_error err);

/**
 * @brief Calls the global error callback with the error
 *
 * This is mainly meant for internal use
 */
void ts_err(ts_error err);

/**
 * @brief Calls `ts_err`, converting the cstr to a `ts_string8`
 *
 * @param err_code Error code
 * @param msg_cstr Error message as a c string
 */
#define TS_ERR(err_code, msg_cstr) ts_err((ts_error){ .code=err_code, .msg=TS_STR8(msg_cstr) })

/**
 * @brief Sets the error callback
 * 
 * This is called by any TurboSpork function that has an error. <br>
 * WARNING: This function will be called from multiple threads.
 * Anything done in this function needs to be threadsafe. <br>
 * The default error callback prints the information to `stderr`
 *
 * @param callback New error callback. Needs to be threadsafe
 */
void ts_err_set_callback(ts_error_callback* callback);

/// Converts a `ts_error_code` to a `ts_string8`. Do not modify the returned string
ts_string8 ts_err_to_str(ts_error_code code);

/// Converst a `ts_string8` to `ts_error_code` 
ts_error_code ts_err_from_str(ts_string8 str);

#endif // ERR_H

