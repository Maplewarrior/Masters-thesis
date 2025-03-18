import matplotlib.pyplot as plt
import torch
import numpy as np
import pdb

class SplineComparer:
    def __init__(self, 
                 N: int,
                 lb: float,
                 ub: float) -> None:
        self.N = N
        self.lb = lb
        self.ub = ub
        self.xs = torch.linspace(start=lb, end=ub, steps=N)

    def calculate_similarity_simulated(self, a1: torch.tensor, b1: torch.tensor,
                             a2: torch.tensor, b2: torch.tensor
                             ):
        """
        A function that calculates the similarity via the inner-product between two affine splines.
        @param a1: The slope of the first spline
        @param b1: The intersect of the first spline
        @param a2: The slope of the second spline
        @param b2: The intersect of the second spline
        """
        dx = (self.ub - self.lb) / self.N
        f = a1 * self.xs + b1
        g = a2 * self.xs + b2
        norm_f = torch.sqrt(torch.sum(f**2 * dx))
        norm_g = torch.sqrt(torch.sum(g**2 * dx))
        return (torch.sum(f * g * dx) / (norm_f * norm_g)).item()

    def calculate_similarity_analytical(self, a1: torch.tensor, b1: torch.tensor,
                             a2: torch.tensor, b2: torch.tensor):
        
        term1 = (a1 * a2) / 3
        term2 = (a1* b2 + a2*b1) / 2
        term3 = b1 * b2

        F_ub = term1 * self.ub**3 + term2 * self.ub**2 + term3 * self.ub
        F_lb = term1 * self.lb**3 + term2 * self.lb**2 + term3 * self.lb

        
        norm_f_ub = (a1**2)/3 * self.ub**3 + a1*b1 * self.ub**2 + b1**2 * self.ub
        norm_f_lb = (a1**2)/3 * self.lb**3 + a1*b1 * self.lb**2 + b1**2 * self.lb
        norm_f = np.sqrt(norm_f_ub - norm_f_lb)


        norm_g_ub = (a2**2)/3 * self.ub**3 + a2*b2 * self.ub**2 + b2**2 * self.ub
        norm_g_lb = (a2**2)/3 * self.lb**3 + a2*b2 * self.lb**2 + b2**2 * self.lb
        norm_g = np.sqrt(norm_g_ub - norm_g_lb)
        
        return (F_ub - F_lb) / (norm_f * norm_g)

    def calculate_simlarity(self, W1, b1, W2, b2):
        similarities = torch.zeros_like((W1))
        for i in range(W1.size(0)):
            for j in range(W1.size(1)):
                similarities[i][j] = self.calculate_similarity_analytical(a1=W1[i][j], b1=b1[i],a2=W2[i][j], b2=b2[i])
                if  similarities[i][j] !=  similarities[i][j]:
                    pdb.set_trace()
                    print("Na value")
        return similarities

        
if __name__ == '__main__':
    N = 100000000
    lb = -10000
    ub =  10000
    SC = SplineComparer(N, lb, ub)

    # define function parameters
    a1 = 0.025
    b1 = 0.3
    a2 = 0.0001
    b2 = -0.3
    
    print(f'a1={a1}, b1={b1}')
    print(f'a2={a2}, b2={b2}')
    # similarity_s = SC.calculate_similarity_simulated(a1=a1, b1=b1,
    #                                      a2=a2, b2=b2)
    # print(f'Similarity simulated: {similarity_s}')

    similarity_a = SC.calculate_similarity_analytical(a1=a1, b1=b1,
                                         a2=a2, b2=b2)
    print(f'Similarity analytical: {similarity_a}')
    
    
    ## NOTE: Variable slope:
    # similarities_a1 = []
    # a1s = []
    # for i in range(250):
    #     a1s.append(a1 + i)
    #     similarities_a1.append(SC.calculate_similarity(a1=a1 + i, b1=b1,
    #                                                    a2=a2, b2=b2))
    
    # plt.scatter(a1s, similarities_a1, s=6, c='red')
    # plt.plot(a1s, similarities_a1, label='sim(f, g)')
    # plt.xlabel('a1')
    # plt.ylabel('Similarity score')
    # plt.title(f'Similarity between f(x)=a1x + {b1} and g(x)={a2}x + {b2} on domain [{lb}, {ub}].')
    # # plt.tight_layout()
    # plt.legend()
    # plt.savefig('function_similarity_plots/similarity_variable_slope1.3.png')

    
    ## NOTE: Variable intersect
    # similarities_b1 = []
    # b1s = []
    # for i in range(2500):
    #     b1s.append(b1 + i)
    #     similarities_b1.append(SC.calculate_similarity(a1=a1, b1=b1 + i,
    #                                                    a2=a2, b2=b2))
    
    # max_sim_b_value = b1s[np.argmax(similarities_b1)]
    # # pdb.set_trace()
    # plt.plot(b1s, similarities_b1, label='sim(f, g)')
    # plt.axvline(x=max_sim_b_value, ymin=0, ymax=np.max(similarities_b1), 
    #             label=f'max similarity, b1={max_sim_b_value}', c='grey', ls='-')
    # plt.xlabel('b1')
    # plt.ylabel('Similarity score')
    # plt.title(f'Similarity between f(x)={a1}x + b1 and g(x)={a2}x + {b2} on domain [{lb}, {ub}].')
    # # plt.tight_layout()
    # plt.legend()
    # plt.savefig('function_similarity_plots/similarity_variable_intersect3.png')

    
    """
    FINDINGS:

    Nearly identical similarity if
    a1, b1 = 0.0, [0, inf]
    a2, b2 = 10, 10
        - Flips completely if we have a negative value of b1
    """


