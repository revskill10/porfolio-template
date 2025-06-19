# Nested Include Example

This file demonstrates **nested includes** - where included content can itself include other content.

## Level 1: This File

This is the top-level markdown file that includes other content.

## Level 2: Including Portfolio Content

{% include "../creative-portfolio/portfolio" %}

## Level 3: Including Blog Content (which may include other content)

{% include "../creative-portfolio/blog" %}

## Recursive Depth

The system handles multiple levels of nesting:

1. **This file** includes portfolio content
2. **Portfolio content** includes individual project details
3. **Project details** may include code files and documentation
4. **Documentation** may include examples and references

## Error Handling

If any level fails to load, the system provides graceful fallbacks:

{% include "./non-existent-file.md" %}

## Performance Considerations

- **Async Loading**: Each include loads independently
- **Caching**: Repeated includes are cached
- **Error Boundaries**: Failed includes don't break the page
- **Loading States**: Users see progress indicators

This demonstrates the power and flexibility of the recursive include system!
