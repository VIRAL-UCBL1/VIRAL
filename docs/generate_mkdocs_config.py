import os
from pathlib import Path
import yaml

def generate_base_config():
    """Generate the base MkDocs configuration."""
    return {
        'site_name': 'VIRAL - Vision-grounded Integration for Reward design And Learning',
        'site_url': 'https://viral-ucbl1.github.io/',
        'repo_url': 'https://github.com/VIRAL-UCBL1/VIRAL',
        'repo_name': 'VIRAL-UCBL1/VIRAL',
        'theme': {
            'name': 'material',
            'icon': {
                'repo': 'fontawesome/brands/github'
            },
            'palette': [
                {
                    'scheme': 'default',
                    'primary': 'indigo',
                    'toggle': {
                        'icon': 'material/brightness-7',
                        'name': 'Switch to dark mode'
                    }
                },
                {
                    'scheme': 'slate',
                    'primary': 'indigo',
                    'toggle': {
                        'icon': 'material/brightness-4',
                        'name': 'Switch to light mode'
                    }
                }
            ],
            'language': 'en',
            'features': [
                'navigation.tabs',
                'search.highlight',
                'search.suggest',
                'search.share',
                'content.code.copy',
                'content.code.prettify',
                'content.action.edit',
                'content.action.view',
                'content.code.annotate',
                'content.tabs.link',
                'navigation.sections'
            ]
        },
        'plugins': [
            'search',
            'autorefs',
            {
                'mkdocstrings': {
                    'handlers': {
                        'python': {
                            'options': {
                                'docstring_style': 'google',
                                'show_signature': True,
                                'show_source': False
                            }
                        }
                    }
                }
            }
        ]
    }

def generate_nav_section():
    """Generate the navigation section based on the docs directory structure."""
    docs_dir = Path('.')
    code_docs_dir = docs_dir / 'code_docs'
    
    nav = [
        {'Home': 'README.md'},
        {'Setup': 'setup.md'}
    ]
    
    doc_section = {'Documentation': []}
    if code_docs_dir.exists():
        # Add main files first
        main_files = ['main.md', 'VIRAL.md']
        for file in main_files:
            if (code_docs_dir / file).exists():
                doc_section['Documentation'].append({file[:-3]: f'code_docs/{file}'})
        
        subdirs = [d for d in code_docs_dir.iterdir() if d.is_dir()]
        for subdir in sorted(subdirs):
            subdir_nav = {subdir.name: []}
            for file in sorted(subdir.glob('*.md')):
                if file.name != '__init__.md':
                    subdir_nav[subdir.name].append(
                        {file.stem: f'code_docs/{subdir.name}/{file.name}'}
                    )
            if subdir_nav[subdir.name]:
                doc_section['Documentation'].append(subdir_nav)
    
    nav.append(doc_section)
    
    if (Path('../LICENSE').exists()):
        nav.append({'License': 'LICENSE.md'})
    
    return nav

def generate_mkdocs_yaml():
    """Generate the complete MkDocs YAML configuration."""
    config = generate_base_config()
    config['nav'] = generate_nav_section()
    
    class CustomDumper(yaml.SafeDumper):
        pass
    
    def str_presenter(dumper, data):
        if '\n' in data:
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)
    
    CustomDumper.add_representer(str, str_presenter)
    
    # Write the configuration to mkdocs.yml
    with open('../mkdocs.yml', 'w') as f:
        yaml.dump(config, f, Dumper=CustomDumper, sort_keys=False, allow_unicode=True)
        print("Generated mkdocs.yml successfully!")

if __name__ == "__main__":
    generate_mkdocs_yaml()