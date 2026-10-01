"""
Shared input utilities for all agents.

Provides multiline input with full editing support.
"""

import sys


def get_multiline_input(prompt: str) -> str:
    """Read input with full editing support.
    
    Features:
    - Paste multi-line content
    - Arrow keys to move cursor anywhere
    - Cmd+Left/Right (Ctrl+A/E) to jump to start/end of line (wraps to prev/next line)
    - Option+Left/Right to jump by word (wraps to prev/next line)
    - Insert/delete characters at cursor position
    - Backspace and Delete keys
    - Enter to submit (after brief pause to allow paste)
    """
    import tty
    import termios
    import os
    import select
    
    print(prompt, end="", flush=True)
    
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    try:
        tty.setcbreak(fd)
        
        lines = ['']  # Buffer as list of lines
        row, col = 0, 0  # Cursor position (line index, column index)
        prompt_len = len(prompt)
        
        def redraw_from_cursor():
            """Redraw from cursor to end, then restore cursor position."""
            save_pos = '\x1b[s'
            clear_to_end = '\x1b[J'
            
            content = lines[row][col:]
            for i in range(row + 1, len(lines)):
                content += '\n' + lines[i]
            
            sys.stdout.write(save_pos + clear_to_end + content + '\x1b[u')
            sys.stdout.flush()
        
        def move_cursor_to(new_row, new_col):
            """Move cursor to new position with screen updates."""
            nonlocal row, col
            
            if new_row < 0 or new_row >= len(lines):
                return
            new_col = max(0, min(new_col, len(lines[new_row])))
            
            row_diff = new_row - row
            if row_diff < 0:
                sys.stdout.write(f'\x1b[{-row_diff}A')
            elif row_diff > 0:
                sys.stdout.write(f'\x1b[{row_diff}B')
            
            screen_col = (prompt_len + new_col) if new_row == 0 else new_col
            sys.stdout.write(f'\x1b[{screen_col + 1}G')
            sys.stdout.flush()
            
            row, col = new_row, new_col
        
        def find_word_boundary_left():
            """Find position of previous word boundary."""
            if col == 0:
                return 0
            line = lines[row]
            pos = col - 1
            # Skip whitespace
            while pos > 0 and line[pos - 1] in ' \t':
                pos -= 1
            # Skip word characters
            while pos > 0 and line[pos - 1] not in ' \t':
                pos -= 1
            return pos
        
        def find_word_boundary_right():
            """Find position of next word boundary."""
            line = lines[row]
            if col >= len(line):
                return len(line)
            pos = col
            # Skip current word
            while pos < len(line) and line[pos] not in ' \t':
                pos += 1
            # Skip whitespace
            while pos < len(line) and line[pos] in ' \t':
                pos += 1
            return pos
        
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], None)
            
            if ready:
                data = os.read(fd, 4096).decode('utf-8', errors='replace')
                if not data:
                    break
                
                i = 0
                while i < len(data):
                    char = data[i]
                    
                    # Check for escape sequences
                    if char == '\x1b' and i + 1 < len(data):
                        remaining = data[i:]
                        
                        # Option+Left (word back) - \x1b[1;3D or \x1bb
                        if remaining.startswith('\x1b[1;3D') or remaining.startswith('\x1bb'):
                            new_pos = find_word_boundary_left()
                            if new_pos == 0 and col == 0 and row > 0:
                                # Already at start, wrap to end of previous line
                                move_cursor_to(row - 1, len(lines[row - 1]))
                            else:
                                move_cursor_to(row, new_pos)
                            i += 6 if remaining.startswith('\x1b[1;3D') else 2
                            continue
                        
                        # Option+Right (word forward) - \x1b[1;3C or \x1bf
                        if remaining.startswith('\x1b[1;3C') or remaining.startswith('\x1bf'):
                            new_pos = find_word_boundary_right()
                            if new_pos == len(lines[row]) and col == len(lines[row]) and row < len(lines) - 1:
                                # Already at end, wrap to start of next line
                                move_cursor_to(row + 1, 0)
                            else:
                                move_cursor_to(row, new_pos)
                            i += 6 if remaining.startswith('\x1b[1;3C') else 2
                            continue
                        
                        # Cmd+Left (start of line) - \x1b[1;2D or \x1b[H or \x1bOH or \x01
                        if remaining.startswith('\x1b[1;2D') or remaining.startswith('\x1b[H') or remaining.startswith('\x1bOH'):
                            if col == 0 and row > 0:
                                # Already at start, wrap to start of previous line
                                move_cursor_to(row - 1, 0)
                            else:
                                move_cursor_to(row, 0)
                            i += 6 if remaining.startswith('\x1b[1;2D') else (3 if remaining.startswith('\x1bOH') else 3)
                            continue
                        
                        # Cmd+Right (end of line) - \x1b[1;2C or \x1b[F or \x1bOF or \x05
                        if remaining.startswith('\x1b[1;2C') or remaining.startswith('\x1b[F') or remaining.startswith('\x1bOF'):
                            if col == len(lines[row]) and row < len(lines) - 1:
                                # Already at end, wrap to end of next line
                                move_cursor_to(row + 1, len(lines[row + 1]))
                            else:
                                move_cursor_to(row, len(lines[row]))
                            i += 6 if remaining.startswith('\x1b[1;2C') else (3 if remaining.startswith('\x1bOF') else 3)
                            continue
                        
                        # Home key - \x1b[1~ or \x1b[H
                        if remaining.startswith('\x1b[1~'):
                            if col == 0 and row > 0:
                                move_cursor_to(row - 1, 0)
                            else:
                                move_cursor_to(row, 0)
                            i += 4
                            continue
                        
                        # End key - \x1b[4~ or \x1b[F
                        if remaining.startswith('\x1b[4~'):
                            if col == len(lines[row]) and row < len(lines) - 1:
                                move_cursor_to(row + 1, len(lines[row + 1]))
                            else:
                                move_cursor_to(row, len(lines[row]))
                            i += 4
                            continue
                        
                        # Standard arrow keys
                        if remaining.startswith('\x1b['):
                            if len(remaining) >= 3:
                                seq = remaining[2]
                                if seq == 'A':  # Up
                                    move_cursor_to(row - 1, col)
                                    i += 3
                                    continue
                                elif seq == 'B':  # Down
                                    move_cursor_to(row + 1, col)
                                    i += 3
                                    continue
                                elif seq == 'C':  # Right
                                    if col < len(lines[row]):
                                        move_cursor_to(row, col + 1)
                                    elif row < len(lines) - 1:
                                        move_cursor_to(row + 1, 0)
                                    i += 3
                                    continue
                                elif seq == 'D':  # Left
                                    if col > 0:
                                        move_cursor_to(row, col - 1)
                                    elif row > 0:
                                        move_cursor_to(row - 1, len(lines[row - 1]))
                                    i += 3
                                    continue
                                elif seq == '3' and len(remaining) >= 4 and remaining[3] == '~':  # Delete
                                    if col < len(lines[row]):
                                        lines[row] = lines[row][:col] + lines[row][col + 1:]
                                        redraw_from_cursor()
                                    elif row < len(lines) - 1:
                                        lines[row] += lines[row + 1]
                                        lines.pop(row + 1)
                                        redraw_from_cursor()
                                    i += 4
                                    continue
                    
                    # Ctrl+A (start of line) - also Cmd+Left on macOS
                    if char == '\x01':
                        if col == 0 and row > 0:
                            # Already at start, wrap to start of previous line
                            move_cursor_to(row - 1, 0)
                        else:
                            move_cursor_to(row, 0)
                        i += 1
                        continue
                    
                    # Ctrl+E (end of line) - also Cmd+Right on macOS
                    if char == '\x05':
                        if col == len(lines[row]) and row < len(lines) - 1:
                            # Already at end, wrap to end of next line
                            move_cursor_to(row + 1, len(lines[row + 1]))
                        else:
                            move_cursor_to(row, len(lines[row]))
                        i += 1
                        continue
                    
                    # Backspace
                    if char in ('\x7f', '\x08'):
                        if col > 0:
                            lines[row] = lines[row][:col - 1] + lines[row][col:]
                            col -= 1
                            sys.stdout.write('\b')
                            redraw_from_cursor()
                        elif row > 0:
                            prev_len = len(lines[row - 1])
                            lines[row - 1] += lines[row]
                            lines.pop(row)
                            row -= 1
                            col = prev_len
                            sys.stdout.write(f'\x1b[A\x1b[{(prompt_len if row == 0 else 0) + col + 1}G')
                            redraw_from_cursor()
                        i += 1
                        continue
                    
                    # Enter/newline
                    if char in ('\n', '\r'):
                        rest = lines[row][col:]
                        lines[row] = lines[row][:col]
                        lines.insert(row + 1, rest)
                        row += 1
                        col = 0
                        sys.stdout.write('\n')
                        redraw_from_cursor()
                        i += 1
                        continue
                    
                    # Regular character - insert at cursor
                    if char >= ' ' or char == '\t':
                        lines[row] = lines[row][:col] + char + lines[row][col:]
                        col += 1
                        sys.stdout.write(char)
                        if col < len(lines[row]):
                            redraw_from_cursor()
                        sys.stdout.flush()
                    
                    i += 1
                
                # Check if we should submit
                if '\n' in data or '\r' in data:
                    more_ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if not more_ready:
                        break
        
        result = '\n'.join(lines).strip()
        if not result.endswith('\n'):
            print()
        return result
        
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
