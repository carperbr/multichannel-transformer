import argparse
import os
import re

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file_exts', type=str, default='m4a,mp4,webm,wav,mp3')
    p.add_argument('--dir', type=str, default='./')
    p.add_argument('--remove_regex', type=str, default='')
    p.add_argument('--append', type=str, default='AI Instrumental')
    args = p.parse_args()

    args.file_exts = [ext for ext in args.file_exts.split(',')]

    files = os.listdir(args.dir)
    dir_name = os.path.basename(os.path.dirname(os.path.join(args.dir, '')))
    
    for i, _ in enumerate(files):
        base_name = os.path.basename(files[i])[::-1]
        idx = base_name.find('.')
        ext = base_name[:idx][::-1]
        base = base_name[idx+1:][::-1]

        base = re.sub(r"(\d+) - ", r"\1 " , base)

        if (ext) in args.file_exts:
            new_filename = f"{dir_name} -{args.append}- {base}.{ext}".replace('-', '—')
            #print(new_filename)
            #print(f'{args.dir}//{new_filename}')
            os.rename(f'{os.path.join(args.dir)}//{files[i]}', f'{os.path.join(args.dir)}//{new_filename}')
    
if __name__ == '__main__':
    main()
    