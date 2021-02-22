module RKF
    function coeff_RKF()
        # Runge–Kutta–Fehlberg method 
        # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method

        # Formula 2, Fehlberg
        n_K = 6
        n_L = 5
        A_rk = zeros(n_K)
        B_rk = zeros(n_K, n_L)
        Ck_rk = zeros(n_K)
        CH_rk = zeros(n_K)
        CT_rk = zeros(n_K)

        # A_rk
        A_rk[2] = 1/4
        A_rk[3] = 3/8
        A_rk[4] = 12/13
        A_rk[5] = 1
        A_rk[6] = 1/2

        # B_rk
        B_rk[2,1] = 1/4
        B_rk[3,1], B_rk[3,2] = 3/32, 9/32  
        B_rk[4,1], B_rk[4,2], B_rk[4,3] = 1932/2197,-7200/2197,7296/2197
        B_rk[5,1], B_rk[5,2], B_rk[5,3], B_rk[5,4] = 439/216,-8,3680/513,-845/4104
        B_rk[6,1], B_rk[6,2], B_rk[6,3], B_rk[6,4], B_rk[6,5] = -8/27, 2, -3544/2565, 1859/4104, -11/40

        # Ck_rk
        Ck_rk[1] = 1/9
        Ck_rk[2] = 0
        Ck_rk[3] = 9/20
        Ck_rk[4] = 16/45
        Ck_rk[5] = 1/12
        Ck_rk[6] = 0

        # CH_rk
        CH_rk[1] = 	16/135  
        CH_rk[2] =  0
        CH_rk[3] = 6656/12825
        CH_rk[4] = 28561/56430
        CH_rk[5] = -9/50 
        CH_rk[6] = 2/55 

        # CT_rk
        CT_rk[1] = 1/360 
        CT_rk[2] = 0
        CT_rk[3] = -128/4275 
        CT_rk[4] = -2187/75240
        CT_rk[5] = 1/50
        CT_rk[6] =	2/55

        return A_rk, B_rk, Ck_rk, CH_rk, CT_rk
    end
end
