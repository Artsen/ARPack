from __future__ import annotations
import sys
import traceback

def main():
    if len(sys.argv) < 2:
        print('', end='')
        return
    path = sys.argv[1]
    try:
        try:
            import fitz
            doc = fitz.open(path)
            parts = [p.get_text('text') for p in doc]
            txt = '\n'.join(parts)
            if txt and txt.strip():
                sys.stdout.write(txt)
                return
        except Exception:
            pass

        try:
            from pdfminer.high_level import extract_text
            txt = extract_text(path) or ''
            if txt and txt.strip():
                sys.stdout.write(txt)
                return
        except Exception:
            pass

    except Exception:
        traceback.print_exc(file=sys.stderr)

if __name__ == '__main__':
    main()
