Consider the matrix 

$$
A_c = \begin{bmatrix} 
1 & c & c & \cdots \\
c & 1 & c & \cdots \\
\vdots & \vdots & & \\
c & c & \cdots & 1
\end{bmatrix}.
$$

For this special case, we have 

$$
R = I - \frac{1}{n}A_c \ \text{ and } \ C = (I - L_c)^{-1}L_c^T.
$$

The actual rate of RCD is defined as follows 

$$
\text{PRate(RCD)} \coloneqq \frac{n}{k}\lim_{k \to \infty}\log ( \frac{\| x^k - x^* \|}{\| x_0 - x^* \|}).
$$

### Deriving Rate(CCD)

With $D = I$ and $-L = \begin{pmatrix}0 & 0 \\ c & 0\end{pmatrix}$, we have 
 
$$D - L = I - L = \begin{pmatrix} 1 & 0 \\ c & 1 \end{pmatrix}.$$
 
Inverting a $2\times 2$ lower-triangular matrix,
 
$$(I - L)^{-1} = \begin{pmatrix} 1 & 0 \\ -c & 1 \end{pmatrix}$$
 
And $L^T = \begin{pmatrix} 0 & -c \\ 0 & 0 \end{pmatrix}$, so
 
$$C = (I-L)^{-1}L^T = \begin{pmatrix} 1 & 0 \\ -c & 1 \end{pmatrix}\begin{pmatrix} 0 & -c \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 0 & -c \\ 0 & c^2 \end{pmatrix}$$
 

Since $C$ is upper triangular, the eigenvalues are the diagonal entries: $0$ and $c^2$. Therefore
 
$$\rho(C) = c^2 \implies \text{Rate(CCD)} = -\log\rho(C) = -\log c^2 = -2\log c.$$

### Deriving ERate(RCD)
 
Since $c < 1$, we have $-\log c > 0$, so this rate is positive. For $n = 2$ and $D = I$,
 
$$R = \Bigl(I - \tfrac{1}{2}A_c\Bigr)^2$$

 
First compute $I - \tfrac{1}{2}A_c$:
 
$$I - \frac{1}{2}A_c = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} - \frac{1}{2}\begin{pmatrix} 1 & c \\ c & 1 \end{pmatrix} = \begin{pmatrix} 1/2 & -c/2 \\ -c/2 & 1/2 \end{pmatrix}$$
 
Thus, 

$$R = \begin{pmatrix}1/2 & -c/2 \\ -c/2 & 1/2\end{pmatrix}.$$
 

Observe that $R$ is symmetric, so eigenvalues come from $\det(R - \lambda I) = 0$:
 
$$\det\begin{pmatrix} 1/2 - \lambda & -c/2 \\ -c/2 & 1/2 - \lambda \end{pmatrix} = (1/2 - \lambda)^2 - \frac{c^2}{4} = 0$$
 
$$\implies \lambda = \frac{1}{2} \pm \frac{c}{2}$$
 
So the eigenvalues are $\frac{1+c}{2}$ and $\frac{1-c}{2}$. Since $c > 0$, the spectral radius is
 
$$\rho(R) = \frac{1+c}{2} \implies \text{ERate(RCD)} = -2\log\rho(R) = -2\log\frac{1+c}{2}$$
 
where the factor of $2$ accounts for the epoch scaling ($n = 2$ steps per epoch).
 
### Deriving PRate(RCD) 
 

Each RCD step updates one coordinate at a time. For a $2\times 2$ matrix, updating coordinate $1$ or coordinate $2$ corresponds to left-multiplying by one of two matrices. From the CD update rule $x^{k+1} = x^k - \tfrac{1}{A_{ii}}A_i x^k e_i$, the two update matrices are:
 
**Update coordinate 1**:
 
$$x \mapsto x - \frac{1}{1}(A_1 x)e_1 = x - \begin{pmatrix}A_{11}x_1 + cx_2 \\ 0\end{pmatrix} = \begin{pmatrix}0 & -c \\ 0 & 1\end{pmatrix}x$$
 
$$A_1 = \begin{pmatrix} 0 & -c \\ 0 & 1 \end{pmatrix}$$
 
**Update coordinate 2**:
 
$$x \mapsto x - \frac{1}{1}(A_2 x)e_2 = x - \begin{pmatrix}0 \\ cx_1 + x_2\end{pmatrix} = \begin{pmatrix}1 & 0 \\ -c & 0\end{pmatrix}x$$
 
$$A_2 = \begin{pmatrix} 1 & 0 \\ -c & 0 \end{pmatrix}$$

 
#### Sample Path Structure
 
In RCD, at each step we independently pick coordinate $1$ (apply $A_1$) or coordinate $2$ (apply $A_2$), each with probability $1/2$. A sample path of $k$ steps looks like a random product of $A_1$'s and $A_2$'s, for example:
 
$$x^k = \cdots A_2 A_1 A_2 A_2 A_1 A_1 A_2 A_1 \, x^0$$
 
Because of the idempotent property $A_i^2 = A_i$, any consecutive repeated matrix collapses:
 
$$A_i A_i = A_i, \quad A_i A_i A_i = A_i, \quad \ldots$$
 
The path therefore reduces to blocks of $A_1 A_2$ or $A_2 A_1$ alternations. Thus, we have
 
$$x^k = (A_1 A_2)^{L_k} x^0 \quad \text{or} \quad x^k = A_2(A_1 A_2)^{L_k} x^0$$
 
where $L_k$ is the random variable counting the number of complete $A_1 A_2$ alternations up to step $k$. Computing $A_1A_2$ gives us,

$$A_1 A_2 = \begin{pmatrix}0 & -c \\ 0 & 1\end{pmatrix}\begin{pmatrix}1 & 0 \\ -c & 0\end{pmatrix} = \begin{pmatrix}0\cdot1+(-c)(-c) & 0 \\ 0\cdot1+1\cdot(-c) & 0\end{pmatrix} = \begin{pmatrix}c^2 & 0 \\ -c & 0\end{pmatrix}$$
 
The matrix is lower triangular, so the eigenvalues are the diagonal entries: $c^2$ and $0$. Thus
 
$$\rho(A_1 A_2) = c^2.$$
 
Each step is independently either $A_1$ or $A_2$ with probability $1/2$ each. An alternation $A_1 A_2$ requires seeing $A_1$ followed by $A_2$, which happens with probability $1/4$ per pair of steps. Over $k$ steps, by the Law of Large Numbers, we have $L_k \approx \frac{k}{4} \ \text{as } k \to \infty$. Therefore,
 
$$\|x^k\| \approx \|(A_1 A_2)^{L_k}\| \|x^0\| \approx \rho(A_1 A_2)^{k/4} \|x^0\| = (c^2)^{k/4}\|x^0\|$$
 
Taking the log and normalizing gives the sample-path rate, 
 
$$\text{PRate(RCD)} = -\frac{2}{k}\log\|(A_1 A_2)^{L_k}\| \approx -\frac{2}{k} \cdot \frac{k}{4}\log(c^2) = -\frac{\log c^2}{2} = -\log c$$

 
### Comparing Rates
 

Thus, we have that $\text{ERate(RCD)} < \text{PRate(RCD)} < \text{Rate(CCD)}$ for all $c\in (0,1)$.
 
 
### Extension to $n \times n$
 
The idempotent property $A_i^2 = A_i$ holds for all $1 \leq i \leq n$ in the general case, by the same argument: applying the same coordinate minimization twice leaves $x$ unchanged. 
