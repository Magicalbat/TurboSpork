/**
 * @file str.h
 * @brief Length based strings and string handling functions (based on Mr 4th's length based strings)
 *
 * This is heavily based on the string header in the [Mr 4th programming series](https://www.youtube.com/@Mr4thProgramming)
 */

#ifndef TS_STR_H
#define TS_STR_H

#include <stdarg.h>

#include "base_defs.h"
#include "mg/mg_arena.h"

/**
 * @brief Length based 8-bit string
 * 
 * Null characters are never included
 */
typedef struct {
    /// Length of string
    ts_u64 size;
    /// Pointer to string characters
    ts_u8* str;
} ts_string8;

/**
 * @brief Node of `ts_string8_list`
 */
typedef struct ts_string8_node {
    struct ts_string8_node* next;
    ts_string8 str;
} ts_string8_node;

/**
 * @brief `ts_string8` singly linked list
 */
typedef struct {
    ts_string8_node* first;
    ts_string8_node* last;

    /// NUmber of nodes
    ts_u64 node_count;
    /// Length of all strings combined
    ts_u64 total_size;
} ts_string8_list; 

/**
 * @brief Creates a `ts_string8` from a string literal
 *
 * Ex: `ts_string8 literal = TS_STR8("Hello World");`
 */
#define TS_STR8(s) ((ts_string8){ sizeof(s)-1, (ts_u8*)s })

/**
 * @brief Creates a `ts_string8` from the pointer range
 * 
 * Does not copy the memory
 */
ts_string8 ts_str8_from_range(ts_u8* start, ts_u8* end);
/**
 * @brief Creates a `ts_string8` from the c string
 * 
 * Does not copy the memory
 */
ts_string8 ts_str8_from_cstr(ts_u8* cstr);

/// Copies a `ts_string8`
ts_string8 ts_str8_copy(mg_arena* arena, ts_string8 str);
/// Creates a c string from a `ts_string8`
ts_u8* ts_str8_to_cstr(mg_arena* arena, ts_string8 str);

/// Returns true if `a` and `b` are equal
ts_b32 ts_str8_equals(ts_string8 a, ts_string8 b);
/// Returns true if `sub` appears in `str`
ts_b32 ts_str8_contains(ts_string8 str, ts_string8 sub);
/// Returns true if `c` appears `str`
ts_b32 ts_str8_contains_char(ts_string8 str, ts_u8 c);

/**
 * @brief Gets the index of the first occurrence of `sub` in `str`
 *
 * @param index Index of the first occurrence of `sub` in `str`
 *  is put in this pointer if an occurrence exists
 *
 * @return true if `sub` is in `str`, false otherwise
 */
ts_b32 ts_str8_index_of(ts_string8 str, ts_string8 sub, ts_u64* index);
/**
 * @brief Gets the index of the first occurrence of `c` in `str`
 *
 * @param index Index of the first occurrence of `c` in `str`
 *  is put in this pointer if an occurrence exists
 *
 * @return true if `c` is in `str`, false otherwise
 */
ts_b32 ts_str8_index_of_char(ts_string8 str, ts_u8 c, ts_u64* index);

/// Creates a `ts_string8` that points to the substring in `str` (does not copy memory)
ts_string8 ts_str8_substr(ts_string8 str, ts_u64 start, ts_u64 end);
/// Creates a `ts_string8` that points to the substring in `str` (does not copy memory)
ts_string8 ts_str8_substr_size(ts_string8 str, ts_u64 start, ts_u64 size);

/// Creates a new `ts_string8` without any occurrences of  ' ', '\t', '\n', and '\r'
ts_string8 ts_str8_remove_space(mg_arena* arena, ts_string8 str);

/**
 * @brief Pushes a `ts_string8` to the `ts_string8_list` with an already allocated node
 *
 * @param list String list to push to
 * @param str String to push
 * @param node Allocated node that is being pushed
 */
void ts_str8_list_push_existing(ts_string8_list* list, ts_string8 str, ts_string8_node* node);
/// Pushes a `ts_string8` to the `ts_string8_list` while allocating the node
void ts_str8_list_push(mg_arena* arena, ts_string8_list* list, ts_string8 str);

/// Creates a string that joins together all of the strings in `list`
ts_string8 ts_str8_concat(mg_arena* arena, ts_string8_list list);

/**
 * @brief Creates a formated `ts_string8` from a c string and a `va_list`
 *
 * Formats string according to the c string format system (i.e. printf) <br>
 * See `ts_str8_pushf` for not `va_list` version
 *
 * @param arena Arena to allocate `ts_string8` on
 * @param fmt C string with specifying the format (e.g. `"Num: %u"`)
 * @param args List of arguments for format
 */
ts_string8 ts_str8_pushfv(mg_arena* arena, const char* fmt, va_list args);
/**
 * @brief Creates a formated `ts_string8` from a c string and a list of arguments
 *
 * Formats string according to the c string format system (i.e. printf) <br>
 *
 * @param arena Arena to allocate `ts_string8` on
 * @param fmt C string with specifying the format (e.g. `"Num: %u"`)
 */
ts_string8 ts_str8_pushf(mg_arena* arena, const char* fmt, ...);

#endif // TS_STR_H

