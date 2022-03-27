# Criminisi Inpainter
Implementation of the inpainting algorithm proposed by [Criminisi et al. (2004)](https://www.irisa.fr/vista/Papers/2004_ip_criminisi.pdf)

Initially, this started as translating [this code](https://github.com/ikuwow/inpainting_criminisi2004) from Matlab to Python. 
However, after working with that code it became apparent that it has a few issues, although it still was helpful.
I eventually found [this other code](https://github.com/igorcmoura/inpaint-object-remover) 
which was referenced to complete my implementation of Criminisi.

Most implementations of this algorithm that I have looked at, as well as my own, don't seem able to exactly replicate Criminisi's results. 
Furthermore, there are some key details on implementation that were left out of the paper, which exacerbates that issue. 
Because of this, I contacted [Antonio Criminisi on LinkedIn](https://www.linkedin.com/in/antonio-criminisi-5ba41aa/?originalSubdomain=uk), 
and asked if he would be willing to share his test images and source code. He said that since this paper was published so long ago he no longer has the code. 
