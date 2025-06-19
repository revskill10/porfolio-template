# Example Content

This is an example markdown file that gets included by other content.

## Features Demonstrated

- **Markdown rendering** with full GitHub Flavored Markdown support
- **Mathematical expressions**: $E = mc^2$ and $$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$
- **Code highlighting**:

```javascript
function fibonacci(n) {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}
```

## Tables

| Feature | Status | Notes |
|---------|--------|-------|
| Includes | ✅ | Working perfectly |
| Math | ✅ | KaTeX rendering |
| Code | ✅ | Syntax highlighting |
| Tables | ✅ | GitHub Flavored Markdown |

## Task Lists

- [x] Implement basic includes
- [x] Add recursive support
- [x] Handle error cases
- [ ] Add caching optimization
- [ ] Performance monitoring

> **Note**: This content is included from `example-content.md` to demonstrate the include system working with various markdown features.

## Nested Include Example

You can even include content that itself includes other content:

{% include "../creative-portfolio/research" %}
