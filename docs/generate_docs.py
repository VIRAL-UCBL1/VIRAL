import os
from pathlib import Path

def generate_docs():
    src_dir = Path('../src')
    docs_dir = Path('code_docs')
    
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    for root, dirs, files in os.walk(src_dir):
        root_path = Path(root)
        
        relative_path = root_path.relative_to(src_dir)
        
        if relative_path != Path('.'):
            doc_dir = docs_dir / relative_path
            doc_dir.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            if file != '__init__.py' and not file.startswith('_'):
                if file.endswith('.py'):
                    if relative_path == Path('.'):
                        module_path = file[:-3]
                    else:
                        module_path = str(relative_path / file[:-3])

                    module_path = module_path.replace(os.sep, '.')
                    
                    md_file = docs_dir / relative_path / f"{file[:-3]}.md"
                    
                    with open(md_file, 'w') as f:
                        f.write(f"::: src.{module_path}\n")
                    
                    print(f"Created documentation file: {md_file}")

if __name__ == "__main__":
    generate_docs()