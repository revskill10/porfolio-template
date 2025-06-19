# Advanced Markdown Features

This content is loaded from an **external markdown file** to demonstrate the file inclusion capabilities of our enhanced markdown widget.

## File Inclusion Benefits

Loading markdown from external files provides several advantages:

1. **Modularity**: Content can be organized into separate files
2. **Reusability**: Same content can be included in multiple places
3. **Maintainability**: Easier to update content without touching HTML
4. **Version Control**: Better tracking of content changes
5. **Collaboration**: Multiple people can work on different content files

## Syntax Highlighting

Here's a JavaScript example with proper syntax highlighting:

```javascript
// Advanced async/await pattern with error handling
async function fetchUserData(userId) {
  try {
    const response = await fetch(`/api/users/${userId}`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const userData = await response.json();
    
    // Transform data
    return {
      id: userData.id,
      name: userData.full_name,
      email: userData.email_address,
      lastLogin: new Date(userData.last_login_timestamp)
    };
  } catch (error) {
    console.error('Failed to fetch user data:', error);
    throw error;
  }
}

// Usage with proper error handling
fetchUserData(123)
  .then(user => console.log('User loaded:', user))
  .catch(error => console.error('Error:', error));
```

## Mathematical Formulas

The **Schrödinger equation** in quantum mechanics:

$$i\hbar\frac{\partial}{\partial t}\Psi(\mathbf{r},t) = \hat{H}\Psi(\mathbf{r},t)$$

Where:
- $\Psi(\mathbf{r},t)$ is the wave function
- $\hat{H}$ is the Hamiltonian operator
- $\hbar$ is the reduced Planck constant
- $i$ is the imaginary unit

## Advanced Tables

| Language | Paradigm | Type System | Performance | Learning Curve |
|----------|----------|-------------|-------------|----------------|
| **Python** | Multi-paradigm | Dynamic | Medium | Easy |
| **Rust** | Systems | Static | High | Steep |
| **JavaScript** | Multi-paradigm | Dynamic | Medium | Easy |
| **Haskell** | Functional | Static | High | Very Steep |
| **Go** | Procedural | Static | High | Medium |
| **TypeScript** | Multi-paradigm | Static | Medium | Medium |

## Code Comparison

### Python Implementation
```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)
```

### Rust Implementation
```rust
fn quicksort<T: Ord + Clone>(arr: &mut [T]) {
    if arr.len() <= 1 {
        return;
    }
    
    let pivot_index = partition(arr);
    let (left, right) = arr.split_at_mut(pivot_index);
    
    quicksort(left);
    quicksort(&mut right[1..]);
}

fn partition<T: Ord>(arr: &mut [T]) -> usize {
    let pivot_index = arr.len() - 1;
    let mut i = 0;
    
    for j in 0..pivot_index {
        if arr[j] <= arr[pivot_index] {
            arr.swap(i, j);
            i += 1;
        }
    }
    
    arr.swap(i, pivot_index);
    i
}
```

## Task Progress

### Completed Features
- [x] Basic markdown parsing
- [x] GitHub Flavored Markdown support
- [x] Mathematical expression rendering
- [x] Syntax highlighting for code blocks
- [x] Table formatting and styling
- [x] External file loading
- [x] Custom component integration

### In Progress
- [ ] Markdoc widget support
- [ ] File inclusion with recursive loading
- [ ] Custom directive processing
- [ ] Performance optimizations

### Planned Features
- [ ] Mermaid diagram support
- [ ] PlantUML integration
- [ ] Interactive code execution
- [ ] Real-time collaborative editing

## Blockquotes and Callouts

> **Important**: This is a standard blockquote that can contain **formatted text** and even inline math like $E = mc^2$.

> **Warning**: Always validate user input when processing markdown content to prevent XSS attacks.

> **Tip**: Use external files for large content blocks to keep your HTML templates clean and maintainable.

## Links and References

- [Markdown Guide](https://www.markdownguide.org/)
- [GitHub Flavored Markdown Spec](https://github.github.com/gfm/)
- [KaTeX Supported Functions](https://katex.org/docs/supported.html)
- [Prism.js Syntax Highlighting](https://prismjs.com/)

---

*This content was loaded from `content/advanced-features.md` to demonstrate external file inclusion capabilities.*
