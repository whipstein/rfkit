import lldb
import re

def get_float_value(val):
    """Extract a display string from a value, handling f32/f64 directly."""
    basic = val.GetValue()
    if basic is not None:
        return basic
    # Fallback: try summary
    summary = val.GetSummary()
    if summary is not None:
        return summary
    return str(val)


def complex_summary(val, internal_dict):
    """
    Summary provider for num_complex::Complex<T>.
    Displays as: (re + im·i), e.g. (1.5 + -3.2·i)
    """
    re_val = val.GetChildMemberWithName("re")
    im_val = val.GetChildMemberWithName("im")

    if not re_val.IsValid() or not im_val.IsValid():
        return None

    re_str = get_float_value(re_val)
    im_str = get_float_value(im_val)

    if re_str is None or im_str is None:
        return None

    # Format imaginary part sign nicely
    try:
        im_float = float(im_str)
        if im_float < 0:
            return f"({re_str} - {abs(im_float)}·i)"
        else:
            return f"({re_str} + {im_str}·i)"
    except ValueError:
        # im_str wasn't a plain float (e.g. NaN, inf) — just display as-is
        return f"({re_str} + {im_str}·i)"


def __lldb_init_module(debugger, internal_dict):
    # Match num_complex::Complex<f32>, num_complex::Complex<f64>, etc.
    debugger.HandleCommand(
        'type summary add '
        '--python-function num_complex_lldb.complex_summary '
        '--expand '
        '-x "^num_complex::Complex<.*>$" '
        '--category Rust'
    )
    # Also match if it appears without the full module path (sometimes LLDB strips it)
    debugger.HandleCommand(
        'type summary add '
        '--python-function num_complex_lldb.complex_summary '
        '--expand '
        '-x "^Complex<.*>$" '
        '--category Rust'
    )
    debugger.HandleCommand('type category enable Rust')
    print("num_complex LLDB formatter loaded.")
