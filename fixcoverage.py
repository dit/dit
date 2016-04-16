import coverage
import re
import sys

def main(argv):
    source = argv[1]
    dest = argv[2]
    cd = coverage.CoverageData()
    cd.read_file('.coverage')

    # Prefilter to filenames in dit
    filenames = [filename for filename in cd._lines.keys()
                 if 'dit' in filename]

    for filename in filenames:
        new_filename = re.sub(source, dest, filename)
        if new_filename != filename:
            cd._lines[new_filename] = cd._lines.pop(filename)

    cd.write_file('.coverage')

if __name__ == '__main__':
    sys.exit(main(sys.argv))
