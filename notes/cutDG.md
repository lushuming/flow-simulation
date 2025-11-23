# cutDG / unfitted DG method

## Cell Agglomeration technique
**Reference:**  
*Johansson, A., Larson, M.G. A high order discontinuous Galerkin Nitsche method for elliptic problems with fictitious boundary. Numer. Math. 123, 607–628 (2013). https://doi.org/10.1007/s00211-012-0497-1*

### ⭐ Core Idea  
The paper develops a high-order unfitted Discontinuous Galerkin (DG) method based on Nitsche’s formulation for elliptic problems defined on fictitious domains. The key technique is to stabilize small–cut elements by associating them with neighboring larger elements.

### 🔧 Method Overview  
- Introduces **extended elements** by merging small–cut cells with adjacent larger cells.  
![alt text](image.png)
- DG basis functions remain easy to define on these extended elements even when their shapes become irregular.  
- Uses this extended partition to naturally construct a stable DG bilinear form via Nitsche’s method.  

### 📐 Theoretical Results  
- **Discrete inverse inequality and trace inequality** can be proved on extended elements, leading to well-posedness.
- The stiffness matrix constructed on extended meshes has a condition number of order  
  $$\kappa(A) = O(h^{-2}),$$
  matching that of standard fitted FEM.  
- Optimal convergence in both $L^2$ and $H^1$ norms is achieved.
- The result is suitable for high order method.

### 🧪 Implementation Notes  
- Stability heavily relies on correctly detecting small–cut elements and merging them (But the choices of large elements have little influence on accuracy and convergence rates). 
- Requires a consistent extension of basis functions across merged elements.

### 🔗 Relevance to My Work  
- Useful for designing stable cutDG / unfitted DG schemes.
- The extended-element strategy might inspire stabilization choices in my EDFM or fracture-flow codes.

### 💭 Personal Comments  
- A clean and classical reference for understanding small–cut stabilization in unfitted DG.  
- The extended-element concept might be more difficult to realize in FEM since the basis functions are hard to properly define.

## Ghost penalty technique
**Reference:**  
*Gürkan, C., Massing, A. A stabilized cut discontinuous Galerkin framework for elliptic boundary value and interface problems. Computer Methods in Applied Mechanics and Engineering, 348, 466-499 (2019).*

### ⭐ Core Idea  


### 🔧 Method Overview  


### 📐 Theoretical Results  


### 🧪 Implementation Notes  


### 🔗 Relevance to My Work  


### 💭 Personal Comments  


