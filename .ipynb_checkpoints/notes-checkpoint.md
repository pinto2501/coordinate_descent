We focus on quadratic optimization problem given by

$$
\min_{x\in\mathbb{R}^n} \frac{1}{2}x^TAx 
$$

where $A$ is a positive definite matrix. Symmetrix matrices with positive eigenvalues are called positive definite. 

## Coordinate Descent (CD) method

Starting with initial point $x_0\in \mathbb{R}^n$, the Coordinate Descent (CD) method, at each iteration $k$ picks a coordiante of $x$, say $i_k$ and update the decision vector by minimizing along the $i_k$th coordiante of the objective function. This gives us,

$$
\begin{aligned}
\frac{d}{dt}\frac{1}{2}(x + te_{i_k})^TA(x + te_{i_k}) &= \frac{1}{2}\frac{d}{dt} (x^T + te_{i_k}^T)A(x+te_{i_k}) \\
&= \frac{1}{2} \frac{d}{dt} (x^TAx + tx^TAe_{i_k} + te_{i_k}^TAx + t^2e_{i_k}^TAe_{i_k}) \\
&= \frac{1}{2} \frac{d}{dt} (2te_{i_k}^TAx + t^2A_{i_k, i_k}) \\
&= e_{i_k}^TAx+ tA_{i_k, i_k} = 0\\ 
\rightarrow t &= - \frac{1}{A_{i_k, i_k}}A_{i_k}x
\end{aligned}
$$

where $A_{i_k}$ represents the $i_k$th row of $A$ and $e_{i_k}$ is the unit vector, whose $i_k$th entry is 1 and the rest of its entries are 0. Thus, the iterative process is 

$$
x^{k+1} = x^{k} - \frac{1}{A_{i_k, i_k}}A_{i_k}x^ke_{i_k}.
$$

### Cyclic Coordinate Descent (CCD) method

When $i_k$ is chosen using the cyclic rule, the algorithm is called the cyclic coordinate descent (CCD) method. The CCD iteration in matrix from can be derived by using the following decomposition,

$$
A = D - L - L^T
$$

where $D$ is the diagonal part of $A$ and $-L$ is the strictly lower triangular part of $A$. By updating coordinate $i_k$, we have

$$
\begin{aligned}
x^{k+1} &= x^{k} - \frac{1}{A_{i_k, i_k}}A_{i_k}x^ke_{i_k} \\
\rightarrow A_{i_k, i_k}x^{k+1} &= A_{i_k, i_k}x^k - A_{i_k}x^ke_{i_k} \\
&= A_{i_k, i_k}x^k - ( \sum_{j < i_k} A_{i_k, j}x^k_j  + A_{i_k, i_k}x^k_{i_k} + \sum_{j > i_k} A_{i_k, j}x^k_j) e_{i_k}
\end{aligned}
$$

Since only the $i_k$th coordinate changes, we have


$$
\begin{aligned}
A_{i_k, i_k}x^{k+1}_{i_k} &= A_{i_k, i_k}x^k_{i_k} - ( \sum_{j < i_k} A_{i_k, j}x^k_j  + A_{i_k, i_k}x^k_{i_k} + \sum_{j > i_k} A_{i_k, j}x^k_j) \\
\rightarrow A_{i_k, i_k}x^{k+1}_{i_k} + \sum_{j < i_k}A_{i_k, j}x^k_j  &= - \sum_{j > i_k} A_{i_k, j}x^k_j.
\end{aligned}
$$

Observe that when we update coordinate $i_k$, the coordinates $j < i_k$ have already been updated in this cycle (so we use $x^{k+1}_j$, while coordinates $j > i_k$ have not yet been updated (so we still use $x^k_j$). This gives us, 

$$
\begin{aligned}
(D - L) x^{k+1} &= L^Tx^k \\
x^{k+1} &= (D + L)^{-1}L^Tx^{k}
\end{aligned}
$$



### Random Coordinate Descent (RCD) method

When $i_k$ is chosen at random among $\{1, \cdots, n\}$ with probabilities $\{p_1, \cdots, p_n\}$ independently at each iteration $k$, the resulting algorithm is called the randomized coordinate descent (RCD) method. Observe that

$$
\begin{aligned}
\mathbb{E}[x^{k+1} | x^k] &= \mathbb{E}[x^{k} - \frac{1}{A_{i, i}}A_{i}x^ke_{i} | x^k] \\
&= x^k - \sum_{i=1}^n p_i \frac{1}{A_{i, i}}A_{i}x^ke_{i}\\
&= x^k - \sum_{i=1}^n p_i \frac{1}{A_{i, i}}e_{i}A_{i}x^k \ \text{ since } A_{i}x^k \in \mathbb{R} \text{ is a scaler }\\
&= (I - \sum_{i=1}^n p_i \frac{1}{A_{i, i}}e_{i}e^T_iA)x^k\\
&= (I - SD^{-1}A)x^k
\end{aligned}
$$

where $S=\text{diag}(p_1,\cdots,p_n)$ and 

$$
\begin{aligned}
\mathbb{E}[x^{\ell(n+1)}_{RCD}] &= \mathbb{E}[ \mathbb{E} [x^{\ell(n+1)}_{RCD} | x^{\ell(n+1) - 1}_{RCD}]]\\
&= (I - SD^{-1}A)^n\ \mathbb{E}[x^{\ell n}_{RCD}] \\
&= R \ \mathbb{E}[x^{\ell n}_{RCD}] 
\end{aligned}
$$

where $R=(I - SD^{-1}A)^n$.

## Asymptotic Rate of Convergence 

$$
\text{Rate(CCD)} \coloneqq \lim_{\ell \to \infty} \sup_{x \in \mathbb{R}^n} -\frac{1}{\ell} \log (\frac{\| x^{\ell n} - x^*\|}{\| x_0 - x^*\|}) = - \log (\rho (C))
$$

and 


$$
\text{Rate(RCD)} \coloneqq \lim_{\ell \to \infty} \sup_{x \in \mathbb{R}^n} -\frac{1}{\ell} \log (\frac{\| E(x^{\ell n})- x^*\|}{\| x_0 - x^*\|}) = - \log (\rho (R))
$$



### Motivating Example 

Consider the $2n \times 2n$ symmetric matrix
 
$$A = I_{2n\times2n} - L - L^T, \qquad L = \frac{1}{n^2}\begin{pmatrix}0_{n\times n} & 0_{n\times n}\\ \mathbf{1}_{n\times n} & 0_{n\times n}\end{pmatrix}$$
 
so that explicitly
 
$$A = \begin{pmatrix}I_{n\times n} & -\dfrac{1}{n^2}\mathbf{1}_{n\times n}\\[6pt] -\dfrac{1}{n^2}\mathbf{1}_{n\times n} & I_{n \times n}\end{pmatrix}_{2n\times 2n}$$
 
#### Eigenvalues of $A$
 
**Eigenvector $v = [\mathbf{1}_{1\times n},\, \mathbf{1}_{1 \times n}]^T$:**
 
Using $\mathbf{1}_{n\times n}\,\mathbf{1}_{1 \times n} = n\mathbf{1}_{1 \times n}$,
 
$$Av = \begin{pmatrix}\mathbf{1}_{1 \times n} - \tfrac{1}{n}\mathbf{1}_{1 \times n} \\ \mathbf{1}_{1 \times n} - \tfrac{1}{n}\mathbf{1}_{1 \times n}\end{pmatrix} = \Bigl(1-\tfrac{1}{n}\Bigr)v$$
 
so $\lambda = 1 - 1/n$ with $\mu := \lambda_{\min}(A) = 1 - 1/n$.
 
**Eigenvector $v = [\mathbf{1}_{1 \times n},\, -\mathbf{1}_{1 \times n}]^T$:**
 
$$Av = \begin{pmatrix}\mathbf{1}_{1 \times n} + \tfrac{1}{n}\mathbf{1}_{1 \times n} \\ -\mathbf{1}_{1 \times n} - \tfrac{1}{n}\mathbf{1}_{1 \times n}\end{pmatrix} = \Bigl(1+\tfrac{1}{n}\Bigr)v$$
 
so $\lambda = 1 + 1/n$.
 
**Remaining $2n-2$ eigenvalues:** For any $v = [u,\, \pm u]^T$ with $u \perp \mathbf{1}_{1\times n}$ (i.e. $\mathbf{1}_{n\times n}\,u = \mathbf{0}$), we get $Av = v$, so $\lambda = 1$ with multiplicity $2n-2$.
 
In summary, the eigenvalues of $A$ are
 
$$\lambda_{\min} = 1-\tfrac{1}{n}, \quad \lambda = 1, \quad \lambda_{\max} = 1+\tfrac{1}{n}.$$
 
#### Eigenvalues of $C = (D-L)^{-1}L^T$
 
Since $D = I_{2n \times 2n}$, we need $(I_{2n \times 2n}-L)^{-1}L^T$. Using block lower-triangular inversion,
 
$$(I_{2n \times 2n} - L)^{-1} = \begin{pmatrix}I_{n\times n} & 0\\ \tfrac{1}{n^2}\mathbf{1}_{n\times n} & I_{n\times n}\end{pmatrix}$$
 
Multiplying out $C = (I_{2n \times 2n}-L)^{-1}L^T$,
 
$$C = \begin{pmatrix}I_{n \times n} & 0\\ \tfrac{1}{n^2}\mathbf{1}_{n\times n} & I_{n \times n}\end{pmatrix} \begin{pmatrix}0 & \tfrac{1}{n^2}\mathbf{1}_{n\times n}\\ 0 & 0\end{pmatrix} = \begin{pmatrix}0_{n\times n} & \tfrac{1}{n^2}\mathbf{1}_{n\times n}\\[4pt] 0_{n\times n} & \tfrac{1}{n^4}\mathbf{1}_{n\times n}\mathbf{1}_{n\times n}\end{pmatrix}$$
 
Using $\mathbf{1}_{n\times n}\mathbf{1}_{n\times n} = n\,\mathbf{1}_{n\times n}$,
 
$$C = \begin{pmatrix}0_{n\times n} & \tfrac{1}{n^2}\mathbf{1}_{n\times n}\\[4pt] 0_{n\times n} & \tfrac{1}{n^3}\mathbf{1}_{n\times n}\end{pmatrix}$$
 
Since every column of $C$ lies in the span of $[{\mathbf{1}_n},\, \tfrac{1}{n}\mathbf{1}_n]^T$, we have $\text{rank}(C) = 1$, so $0$ is an eigenvalue of multiplicity $2n-1$. The unique nonzero eigenvalue equals the trace,
 
$$\text{tr}(C) = 0 + \tfrac{1}{n^3}\,\text{tr}(\mathbf{1}_{n\times n}) = \tfrac{1}{n^3}\cdot n = \tfrac{1}{n^2}.$$

 
Therefore,
 
$$\rho(C) = \frac{1}{n^2}, \text{ and } \ \text{Rate(CCD)} = -\log\rho(C) = 2\log n.$$
 
#### Eigenvalues of $R = (I - \tfrac{1}{n}D^{-1}A)^n$
 
Since $D = I$, this simplifies to $R = (I - \tfrac{1}{n}A)^n$. If $\lambda$ is an eigenvalue of $A$, then $(1 - \lambda/n)^n$ is an eigenvalue of $R$. 

$\textit{Proof}$.

$$
\begin{aligned}
Rv = (I - \frac{1}{n}A)^n v &= (I - \frac{1}{n}A)^{n-1}(v - \frac{1}{n}\lambda v) \\
&= (I - \frac{1}{n}\lambda)^n v.
\end{aligned}
$$

Substituting the eigenvalues of $A$,
 
| Eigenvalue of $A$ | Eigenvalue of $R$ |
|:---:|:---:|
| $1 - 1/n$ | $(1 - \tfrac{1}{n} + \tfrac{1}{n^2})^n$ |
| $1$ | $(1 - \tfrac{1}{n})^n$ |
| $1 + 1/n$ | $(1 - \tfrac{1}{n} - \tfrac{1}{n^2})^n$ |
 
The spectral radius is the largest of these. Since $1 - 1/n < 1 + 1/n$, the first row gives the largest value:
 
$$\rho(R) = \Bigl(1 - \frac{1}{n} + \frac{1}{n^2}\Bigr)^n.$$
 
To bound this from below, apply $(1-x)^n \geq 1 - nx$ with $x = 1/n - 1/n^2$:
 
$$\rho(R) \geq 1 - n\Bigl(\frac{1}{n} - \frac{1}{n^2}\Bigr) = \frac{1}{n} = 1 - \mu$$
 
so $\log\rho(R) \leq \log n$, giving $\text{Rate(RCD)} \leq \log n$.
 
#### Rate Comparison
 
Dividing the two rates,
 
$$\frac{\text{Rate(CCD)}}{\text{Rate(RCD)}} = \frac{2\log n}{-\log\rho(R)} \geq \frac{2\log n}{\log n} = 2, \quad \forall\, n \geq 1.$$
 
CCD is at least twice as fast as RCD on this example for all $n \geq 1$.
 
---



























