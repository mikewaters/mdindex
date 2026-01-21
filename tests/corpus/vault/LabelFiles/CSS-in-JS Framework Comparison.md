# CSS-in-JS Framework Comparison

## Runtime CSS-in-JS Solutions

### Styled Components

**Popularity**: Most popular CSS-in-JS library
**GitHub Stars**: 39.4k
**Created**: 2016

Key Features:

- Template literal syntax

- Automatic critical CSS

- Dynamic styling based on props

- Theming support

- Built-in animation support

```jsx
const Button = styled.button`
  background: ${props => props.primary ? 'blue' : 'gray'};
  color: white;
  padding: 10px 20px;
  
  &:hover {
    background: ${props => props.primary ? 'darkblue' : 'darkgray'};
  }
`

// Usage
<Button primary>Click me</Button>
```

### Emotion

**Popularity**: Second most popular CSS-in-JS solution
**GitHub Stars**: 16.4k
**Created**: 2017

Key Features:

- Multiple styling approaches (styled API, css prop, object styles)

- Framework agnostic

- High performance

- Server-side rendering support

- Zero-config support

```jsx
// Styled API
const Button = styled.button`
  background: ${props => props.primary ? 'blue' : 'gray'};
`

// css prop
<button css={css`
  background: blue;
  &:hover { background: darkblue; }
`}>
  Click me
</button>

// Object styles
const style = css({
  background: 'blue',
  ':hover': {
    background: 'darkblue'
  }
})
```

### JSS

**Popularity**: Popular especially in Material-UI ecosystem
**GitHub Stars**: 7.1k
**Created**: 2014

Key Features:

- Object syntax

- Plugin system

- Framework agnostic

- High performance

- SSR support

```jsx
const styles = {
  button: {
    background: 'blue',
    '&:hover': {
      background: 'darkblue'
    }
  },
  primary: {
    background: 'gold'
  }
}

const useStyles = createUseStyles(styles)
```

### Goober

**Popularity**: Growing, especially for size-conscious applications
**GitHub Stars**: 3.4k
**Created**: 2019

Key Features:

- Tiny size (\~1KB)

- Similar API to styled-components

- Forward refs support

- CSS prop support

```jsx
const Button = styled.button`
  background: ${props => props.primary ? 'blue' : 'gray'};
`
```

## Zero-Runtime CSS-in-JS Solutions

### Linaria

**Popularity**: Leading zero-runtime solution
**GitHub Stars**: 9.8k
**Created**: 2019

Key Features:

- Zero runtime

- CSS extraction at build time

- Dynamic styles without runtime overhead

- TypeScript support

```jsx
const Button = styled.button`
  background: ${props => props.primary ? 'blue' : 'gray'};
`
```

### Vanilla Extract

**Popularity**: Growing rapidly in TypeScript communities
**GitHub Stars**: 7.9k
**Created**: 2021

Key Features:

- Zero runtime

- TypeScript-first

- Static CSS extraction

- Design token support

- Themeing system

```typescript
const styles = styleVariants({
  primary: { background: 'blue' },
  secondary: { background: 'gray' }
});
```

### Compiled

**Popularity**: Newer entrant
**GitHub Stars**: 3.2k
**Created**: 2020

Key Features:

- Zero runtime

- Type-safe styles

- Framework agnostic

- Dynamic styles at build time

```typescript
const button = css`
  background: blue;
  &:hover {
    background: darkblue;
  }
`
```

## Feature Comparison Matrix

| Feature | Styled Components | Emotion | JSS | Goober | Linaria | Vanilla Extract | 
|---|---|---|---|---|---|---|
| Runtime Cost | Yes | Yes | Yes | Yes | No | No | 
| Bundle Size (kb) | \~12.7 | \~7.9 | \~6.2 | \~1 | 0 | 0 | 
| SSR Support | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 
| TypeScript Support | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 
| Dynamic Styles | ✅ | ✅ | ✅ | ✅ | Limited | Limited | 
| Framework Agnostic | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 
| Development Tools | ✅ | ✅ | ✅ | Limited | ✅ | ✅ | 

## Performance Considerations

### Runtime Solutions

Pros:

- Full dynamic capabilities

- Easier debugging

- More flexible

Cons:

- Runtime overhead

- Larger bundle size

- Style recalculation costs

### Zero-Runtime Solutions

Pros:

- No runtime overhead

- Smaller bundle size

- Better performance

Cons:

- Limited dynamic capabilities

- More complex build setup

- Harder debugging

## Usage Recommendations

### Choose Runtime CSS-in-JS when:

- Need highly dynamic styles

- Working on smaller applications

- Prioritize developer experience

- Need full runtime access to theme/props

### Choose Zero-Runtime CSS-in-JS when:

- Performance is critical

- Building larger applications

- Working with static/semi-static styles

- Need smallest possible bundle size

### Framework-Specific Recommendations:

- React + TypeScript: Vanilla Extract or Emotion

- React (JavaScript): Styled Components or Emotion

- Next.js: Linaria or Vanilla Extract

- Size-Critical: Goober

- Material-UI: JSS

- Static Sites: Zero-runtime solutions