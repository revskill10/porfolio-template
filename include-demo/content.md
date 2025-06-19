# Dynamic Include System

This markdown file demonstrates the **powerful include capabilities** of our enhanced markdown renderer.

## Basic Includes

You can include entire portfolio pages:

{% include "../creative-portfolio/about" %}

## Including Components

Individual components work too:

{% include "../creative-portfolio/components/footer" %}

## Including Other Markdown

You can include other markdown files for modular content:

{% include "./example-content.md" %}

## Mathematical Content

Include mathematical blog posts:

{% include "../creative-portfolio/blog/attention-mathematics/content.md" %}

## Code Examples

Include code files directly:

```python
# This could be loaded from an external file
{% include "../creative-portfolio/code/attention.py" %}
```

## Benefits

1. **Modularity**: Content can be organized into reusable pieces
2. **Consistency**: Shared components ensure uniform styling
3. **Maintainability**: Update content in one place, reflect everywhere
4. **Flexibility**: Mix static and dynamic content seamlessly

## Technical Features

- ✅ **Recursive includes** - included content can include other content
- ✅ **Path resolution** - relative paths work correctly
- ✅ **Error handling** - graceful fallbacks for missing content
- ✅ **Performance** - async loading with caching
- ✅ **Context merging** - data from JSON files is properly merged

This system enables powerful content composition while maintaining the simplicity of markdown!
