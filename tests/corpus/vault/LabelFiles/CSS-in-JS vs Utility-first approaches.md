# CSS-in-JS vs Utility-first approaches

CSS-in-JS means you write your styles directly in your JavaScript/TypeScript files, typically using JavaScript template literals or objects. The styles are scoped to specific components and often generated at runtime. Here's an example using Emotion:

```jsx
const Button = styled.button`
  background: blue;
  padding: 8px 16px;
  border-radius: 4px;
  
  &:hover {
    background: darkblue;
  }
`
```

Utility-first CSS, like Tailwind, means you compose styles by applying pre-defined utility classes directly in your HTML/JSX. Instead of writing CSS, you use small, single-purpose classes:

```jsx
<button className="bg-blue-500 px-4 py-2 rounded hover:bg-blue-700">
  Click me
</button>
```

Key differences:

1. **Style Location**

   - CSS-in-JS: Styles are written in JavaScript files, alongside component logic

   - Utility-first: Styles are applied through classes in the markup/JSX

2. **Runtime Behavior**

   - CSS-in-JS: Often generates and injects CSS at runtime (though some solutions like Linaria extract at build time)

   - Utility-first: All classes are pre-defined and available at build time

3. **Style Composition**

   - CSS-in-JS: You compose styles using JavaScript features (template literals, objects, functions)

   - Utility-first: You compose styles by combining utility classes in your markup

4. **Dynamic Styling**

   - CSS-in-JS: Can easily create dynamic styles based on props or state

   ```jsx
   const Button = styled.button`
     background: ${props => props.primary ? 'blue' : 'gray'};
   `
   ```

   - Utility-first: Typically handles dynamic styles through class composition

   ```jsx
   <button className={`${primary ? 'bg-blue-500' : 'bg-gray-500'}`}>
   ```

5. **Bundle Size Impact**

   - CSS-in-JS: Often includes runtime library code and generates styles dynamically

   - Utility-first: Only includes the utility classes that are actually used (when properly configured)

6. **Learning Curve**

   - CSS-in-JS: Requires learning the specific library's API and patterns

   - Utility-first: Requires learning the utility class naming conventions

7. **Developer Experience**

   - CSS-in-JS: 

      - Better TypeScript integration

      - Easier refactoring with component co-location

      - More powerful dynamic styling

   - Utility-first:

      - No context switching between files

      - Faster development with predefined utilities

      - Easier to maintain consistent values

8. **Performance**

   - CSS-in-JS: 

      - Can have runtime overhead

      - May cause style recalculation on prop changes

   - Utility-first:

      - No runtime overhead

      - All styles are pre-compiled

9. **Tooling Support**

   - CSS-in-JS:

      - Great IDE support for JavaScript

      - Harder to use traditional CSS tools

   - Utility-first:

      - Excellent IDE support with plugins

      - Works well with existing CSS tools

Here's a practical example showing both approaches solving the same problem:

CSS-in-JS (Emotion):

```jsx
const Card = styled.div`
  padding: 16px;
  margin: 8px;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  
  ${props => props.highlighted && `
    border: 2px solid blue;
    background: lightblue;
  `}
`

// Usage
<Card highlighted={isSelected}>
  Content here
</Card>
```

Utility-first (Tailwind):

```jsx
const Card = ({ highlighted, children }) => (
  <div className={`
    p-4 m-2 rounded shadow-md
    ${highlighted ? 'border-2 border-blue-500 bg-blue-100' : ''}
  `}>
    {children}
  </div>
)

// Usage
<Card highlighted={isSelected}>
  Content here
</Card>
```

Neither approach is universally better - the choice often depends on:

- Project requirements (runtime vs build-time styles)

- Team preferences

- Performance requirements

- Existing codebase patterns

- Need for dynamic styling

- Build tooling constraints