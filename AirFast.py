# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 16:18:53 2021

@author: 安旭溟
"""

import numpy as np

from scipy.optimize import root_scalar
from scipy.optimize import minimize_scalar
import math
import logging
import matplotlib.pyplot as plt
import time

# python 中的 fractions 类精度也是有限的，可以认为只有当分子分母都为整数时，才是完美精确的
# fractions 类最好不要和其他类（整数类除外）混合进行4则运算
# numpy 中的元素都是假分数, 但是直接访问单个元素还是得到小数。
# np.set_printoptions(formatter={'all': lambda x: str(fractions.Fraction(x).limit_denominator())})
# 上面的语句根本不管用，要逐个添加分数元素才能构成正宗的分数数组
# 设置日志输出模式
t = time.strftime("-%Y-%m-%d-%H-%M-%S", time.localtime())
logging.basicConfig(level=logging.INFO, filename=t + ".log", format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger()

max_for_equation = 6000
tol_for_y_equation = 5.0e-16
tol_for_equation = 2.0e-16
tol_for_min = 2.0e-16
end_point = 1.0e-15


# For the dividing of K_1 and K_2 in line 1
def divide_set(a, B_max, Beta_max, H):
    L_I, L_II = [], []  # 存储都应该是节点的原始索引号
    sum_I = 0
    for i in range(len(H)):
        if a * H[i] * B_max[i] >= Beta_max[i]:
            L_I.append(i)
            sum_I = sum_I + Beta_max[i]
        else:
            L_II.append(i)

    return sum_I, L_I, L_II


# The left side of eq (25)
def Pa_x(B_max, Beta_max, a, H, x):
    sum_I, L_I, L_II = divide_set(a, B_max, Beta_max, H)
    y = 0
    for k in L_II:
        mm = min(a * H[k] * B_max[k] + x, Beta_max[k])
        y = y + mm

    return y + sum_I - 1


# 二分法求根也要分区间，要不然太麻烦了。
# 根的定义域是[0, +oo)，如果不存在根程序可以输出负数，以作指示之用。
# Result return the $\lambda^*$
def Fa_bi(xmin, xmax, stepmax, tol, Beta_max, H, B_max, a):
    lv = Pa_x(B_max, Beta_max, a, H, xmin)
    # print(lv)
    rv = Pa_x(B_max, Beta_max, a, H, xmax)
    # print(rv)
    if abs(lv) < tol_for_y_equation and abs(rv) < tol_for_y_equation:
        lv = 0
    elif abs(lv) < tol_for_y_equation <= abs(rv):
        lv = 0
    elif abs(lv) >= tol_for_y_equation > abs(rv):
        rv = 0
    if lv == 0:
        return xmin
    if rv == 0:
        return xmax
    if lv * rv > 0:
        # print('Error: No root for P(a,x)=1!')
        if lv < 0:
            logger.info("P(a,x) < 0 is unreasonable when a={} and x is in[{}, {}], P(a, xmin)={}, P(a, xmax)={}."
                        .format(a, xmin, xmax, lv, rv))
            return -10000
        else:
            logger.info("P(a,x) > 0: No root for P(a,x)=1! when a={} and x is in [{}, {}], P(a, xmin)={}, P(a,xmax)={}."
                        .format(a, xmin, xmax, lv, rv))
            return -20000

    for step in range(1, stepmax + 1):
        c = (xmin + xmax) / 2
        if Pa_x(B_max, Beta_max, a, H, c) == 0:
            return c
        if abs((xmax - xmin) / 2) < tol:
            return c
        if Pa_x(B_max, Beta_max, a, H, c) * Pa_x(B_max, Beta_max, a, H, xmin) < 0:
            xmax = c
        else:
            xmin = c

    return (xmin + xmax) / 2


def Fa_sci(Beta_max, H, B_max, a, xmin, xmax):
    def f(x):
        p = Pa_x(B_max, Beta_max, a, H, x)
        return p

    sol = root_scalar(f, bracket=[xmin, xmax], method='bisect', xtol=tol_for_equation, maxiter=max_for_equation)

    result = sol.root

    return result


def test_for_Fa_min(Beta_max, H, B_max, a_max):
    A = [0.2 * a_max, 0.4 * a_max, 0.6 * a_max, 0.8 * a_max, 1.0 * a_max, 1.2 * a_max]
    for a in A:
        sum_I, L_I, L_II = divide_set(a, B_max, Beta_max, H)
        X = []
        for i in L_II:
            X.append(Beta_max[i] - a * H[i] * B_max[i])
        xmax = max(X)
        result = Fa_sci(Beta_max, H, B_max, a, 0, 2 * xmax)
        print(result)

    return None


def Hk_a(a, Beta_max, H, B_max, k):
    # k 为节点对应的索引号
    sum_I, L_I, L_II = divide_set(a, B_max, Beta_max, H)
    X = []
    for i in L_II:
        X.append(Beta_max[i] - a * H[i] * B_max[i])
        # X.append((FRa(Beta_max[i] * fra_scal) / fra_scal) -
        #          a * (FRa(H[i] * fra_scal) / fra_scal) * (FRa(B_max[i] * fra_scal) / fra_scal))
    xmax = max(X)
    fa = Fa_bi(0, 2 * xmax, max_for_equation, tol_for_equation, Beta_max, H, B_max, a)
    if fa == -10000 or fa == -20000:
        # print("P(a,x)'s Root Error in func Hk_a when a={}, k={}".format(a, k))
        logger.info("Root Error in func Hk_a when a={}, k={}".format(a, k))
    result = fa - ((Beta_max[k]) - a * (H[k]) * (B_max[k]))

    return result


def bisection_Hk_a(amin, amax, stepmax, tol, Beta_max, H, B_max, k):
    lv = Hk_a(amin, Beta_max, H, B_max, k)
    rv = Hk_a(amax, Beta_max, H, B_max, k)
    if abs(lv) < tol_for_y_equation and abs(rv) < tol_for_y_equation:
        lv = 0
    elif abs(lv) < tol_for_y_equation <= abs(rv):
        lv = 0
    elif abs(lv) >= tol_for_y_equation > abs(rv):
        rv = 0
    if lv == 0:
        return amin
    if rv == 0:
        return amax
    if lv * rv > 0:
        print("Error: No root for H_k(a)=0.")
        logger.info("Error: No root for H_k(a)=0 when k={}.".format(k))
        if lv < 0:
            logger.info("Hk(a) < 0 when k={} and a is in[{}, {}], Hk(amin)={}, Hk(amax)={}."
                        .format(k, amin, amax, lv, rv))
            return -10000
        else:
            logger.info("Hk(a) > 0 when k={} and a is in [{}, {}], Hk(amin)={}, Hk(amax)={}."
                        .format(k, amin, amax, lv, rv))
            return -20000

    for step in range(1, stepmax + 1):
        c = (amin + amax) / 2
        if Hk_a(c, Beta_max, H, B_max, k) == 0:
            return c
        if abs((amax - amin) / 2) < tol:
            return c
        if Hk_a(c, Beta_max, H, B_max, k) * Hk_a(amin, Beta_max, H, B_max, k) < 0:
            amax = c
        else:
            amin = c

    return (amin + amax) / 2


def golden_search_Hk_a(amin, amax, tol, Beta_max, H, B_max, k):
    i = 0
    while True:
        i += 1
        a1 = amax - 0.618 * (amax - amin)
        a2 = amin + 0.618 * (amax - amin)
        if -1 * Hk_a(a1, Beta_max, H, B_max, k) > -1 * Hk_a(a2, Beta_max, H, B_max, k):
            amax = a2
        elif -1 * Hk_a(a1, Beta_max, H, B_max, k) <= -1 * Hk_a(a2, Beta_max, H, B_max, k):
            amin = a1
        DX = abs(amax - amin)
        if DX <= tol:
            result_h = Hk_a((amin + amax) / 2, Beta_max, H, B_max, k)
            result_a = (amin + amax) / 2
            break
        else:
            pass

    return result_h, result_a


def min_sci_Hk_a(amin, amax, tol, Beta_max, H, B_max, k):
    if amin <= amax:
        def Hk(x):
            return Hk_a(x, Beta_max, H, B_max, k)

        res = minimize_scalar(Hk, bounds=(amin, amax), method='bounded',
                              options={'xatol': tol, 'maxiter': max_for_equation, 'disp': 0})
        return res.fun, res.x
    else:
        # logger.info("The interval isn't apporiate.")
        print("The interval isn't apporiate.")
        return None


def test_for_min_sci_Hk_a(Beta_max, H, B_max):
    for k in range(len(H)):
        min_sci_Hk_a(0, 0.02, tol_for_min, Beta_max, H, B_max, k)


# Z is a list storing the intervals of variable s in P18
def gen_z(H, B_max, Beta_max, S):
    a_max, max_index = amax(Beta_max, S)
    S = np.concatenate((S, [a_max]))
    # 去除重复元素
    S = np.unique(S)
    Sn = np.sort(S)[0:max_index + 1].tolist()
    logger.info('The length of Sn is {}.'.format(len(Sn)))
    logger.info("The simple Sn is {}.".format(Sn))
    Z = []
    for i in range(len(Sn) + 1):
        Zi = []
        if i == 0:
            lk = end_point  # 第一个区间左顶点要求开区间，所以加一个微小的值。
            rk = Sn[i] - end_point  # 开区间需要一个极小的数配合端点来模拟.
        elif i != len(Sn):
            lk = Sn[i - 1]
            rk = Sn[i] - end_point
        else:
            lk = Sn[i - 1]
            rk = 2 * Sn[i - 1]

        sum_I, L_I, L_II = divide_set(lk, B_max, Beta_max, H)
        # There we add a if also need else!
        if len(L_II) != 0:
            XL = []
            # XR = []
            sum_bh = 0
            for j in L_II:
                XL.append(Beta_max[j] - lk * H[j] * B_max[j])
                sum_bh = sum_bh + B_max[j] * H[j]
            xml = max(XL)
            a_star = (1 - sum_I) / sum_bh
            # print("a_star is {} and xmax is {} when i={}.".format(a_star, 2 * xml, i))
            logger.info("a_star is {} and xmax is {} when i={}.".format(a_star, 2 * xml, i))

            # 求得的根的精度，也会影响计算结果，此问题待定。
            if lk <= a_star < rk:
                # logger.info("We are going to compute xl.")
                xl = Fa_bi(0, 2 * xml, max_for_equation, tol_for_equation, Beta_max, H, B_max, lk)
                # logger.info("We are going to compute xa.")
                xa = Fa_bi(0, 2 * xml, max_for_equation, tol_for_equation, Beta_max, H, B_max, a_star)
                logger.info("No root in interval: [{}, {}].".format(a_star + end_point, rk))
                Zi.extend([a_star + end_point, rk])
                xr, rk = xa, a_star
            elif rk <= a_star:
                # logger.info("We are going to compute xl.")
                xl = Fa_bi(0, 2 * xml, max_for_equation, tol_for_equation, Beta_max, H, B_max, lk)
                # logger.info("We are going to compute xa.")
                xr = Fa_bi(0, 2 * xml, max_for_equation, tol_for_equation, Beta_max, H, B_max, rk)
            elif a_star < lk:
                logger.info("No root in A_{}: [{}, {}].".format(i, lk, rk))
                Zi.extend([lk, rk])
                continue
            # print for debug the root's exits
            if xl == -1 or xl == -2:
                logger.info("Root Error in left when i = {}, a={} and amax is {}!!".format(i, lk, a_max))
            if xr == -1 or xr == -2:
                logger.info("Root Error in right when i = {}, a={} and amax is {}!!".format(i, rk, a_max))
            # print for debug end

            for k in L_II:
                beta_l = (Beta_max[k]) - lk * (H[k]) * (B_max[k])
                beta_r = (Beta_max[k]) - rk * (H[k]) * (B_max[k])
                if xl < beta_l and xr < beta_r:
                    Zi.extend([lk, rk])
                    # print("For node {} in case I, we get 2 z points: {} and {}.".format(k, lk, rk))
                    logger.info("For node {} in case I, we get 2 z points: {} and {}.".format(k, lk, rk))
                    # print("Node {} in L_IIii when a in [{}, {}].".format(k, lk, rk))
                    logger.info("Node {} in L_IIii when a in [{}, {}].".format(k, lk, rk))
                elif xl >= beta_l and xr < beta_r:
                    logger.info("We are going to compute ap in [{}, {}] for node {}.".format(lk, rk, k))
                    ap = bisection_Hk_a(lk, rk, max_for_equation, tol_for_equation, Beta_max, H, B_max, k)
                    if ap == -1 or ap == -2:
                        # print("No root for Hk(a) for a in [{}, {}] when k={}.".format(lk, rk, k))
                        logger.info("No root for Hk(a) for a in [{}, {}] when k={}.".format(lk, rk, k))
                    Zi.extend({lk, ap, ap + end_point, rk})
                    # print("For node {} in case II, we get 4 z points: {}, {}, {} and {}."
                    #       .format(k, lk, ap, ap + end_point, rk))
                    logger.info("For node {} in case II, we get 4 z points: {}, {}, {} and {}."
                                .format(k, lk, ap, ap + end_point, rk))
                    # print("Node {} in L_IIi when a in [{}, {}].".format(k, lk, ap))
                    logger.info("Node {} in L_IIi when a in [{}, {}].".format(k, lk, ap))
                    # print("Node {} in L_IIii when a in [{}, {}].".format(k, ap + end_point, rk))
                    logger.info("Node {} in L_IIii when a in [{}, {}].".format(k, ap + end_point, rk))
                elif xl < beta_l and xr >= beta_r:
                    logger.info("We are going to compute ap in [{}, {}] for node {}.".format(lk, rk, k))
                    ap = bisection_Hk_a(lk, rk, max_for_equation, tol_for_equation, Beta_max, H, B_max, k)
                    if ap == -1 or ap == -2:
                        # print("No root for Hk(a) for a in [{}, {}] when k={}.".format(lk, rk, k))
                        logger.info("No root for Hk(a) for a in [{}, {}] when k={}.".format(lk, rk, k))
                    Zi.extend({lk, ap - end_point, ap, rk})
                    # print("For node {} in case III, we get 4 z points: {}, {}, {}, {}."
                    #       .format(k, lk, ap - end_point, ap, rk))
                    logger.info("For node {} in case III, we get 4 z points: {}, {}, {}, {}."
                                .format(k, lk, ap - end_point, ap, rk))
                    # print("Node {} in L_IIii when a in [{}, {}].".format(k, lk, ap - end_point))
                    logger.info("Node {} in L_IIii when a in [{}, {}].".format(k, lk, ap - end_point))
                    # print("Node {} in L_IIi when a in [{}, {}].".format(k, ap, rk))
                    logger.info("Node {} in L_IIi when a in [{}, {}].".format(k, ap, rk))
                elif xl >= beta_l and xr >= beta_r:
                    logger.info("We are going to compute r_h and r_a in [{}, {}] for node {}.".format(lk, rk, k))
                    r_h, r_a = golden_search_Hk_a(lk, rk, tol_for_min, Beta_max, H, B_max, k)
                    # r_h, r_a = min_sci_Hk_a(lk, rk, tol_for_min, Beta_max, H, B_max, k)
                    if r_h >= 0:
                        Zi.extend([lk, rk])
                        # print("For node {} in case IV-I, we get 2 z points: {} and {}.".format(k, lk, rk))
                        logger.info("For node {} in case IV-I, we get 2 z points: {} and {}.".format(k, lk, rk))
                        # print("Node {} in L_IIi when a in [{}, {}].".format(k, lk, rk))
                        logger.info("Node {} in L_IIi when a in [{}, {}].".format(k, lk, rk))
                    else:
                        logger.info("We are going to compute ap in [{}, {}] for node {}.".format(lk, r_a, k))
                        al = bisection_Hk_a(lk, r_a, max_for_equation, tol_for_equation, Beta_max, H, B_max, k)
                        if al == -1 or al == -2:
                            # print("No root for Hk(a) for a in [{}, {}] when k={}.".format(lk, r_a, k))
                            logger.info("No root for Hk(a) for a in [{}, {}] when k={}.".format(lk, r_a, k))
                        logger.info("We are going to compute ap in [{}, {}] for node {}.".format(r_a, rk, k))
                        ar = bisection_Hk_a(r_a, rk, max_for_equation, tol_for_equation, Beta_max, H, B_max, k)
                        if ar == -1 or ar == -2:
                            # print("No root for Hk(a) for a in [{}, {}] when k={}.".format(r_a, rk, k))
                            logger.info("No root for Hk(a) for a in [{}, {}] when k={}.".format(r_a, rk, k))
                        Zi.extend([lk, al, al + end_point, ar - end_point, ar, rk])
                        # print("For node {} in case IV-II, we get 6 z points: {}, {}, {}, {}, {} and {}."
                        #       .format(k, lk, al, (al + end_point),
                        #               (ar - end_point), ar, rk))
                        logger.info("For node {} in case IV-II, we get 6 z points: {}, {}, {}, {}, {} and {}."
                                    .format(k, lk, al, (al + end_point),
                                            (ar - end_point), ar, rk))
                        # print("Node {} in L_IIi when a in [{}, {}] and [{}, {}]."
                        #       .format(k, lk, al, ar, rk))
                        logger.info("Node {} in L_IIi when a in [{}, {}] and [{}, {}]."
                                    .format(k, lk, al, ar, rk))
                        # print("Node {} in L_IIii when a in [{}, {}]."
                        #       .format(k, (al + end_point), (ar - end_point)))
                        logger.info("Node {} in L_IIii when a in [{}, {}]."
                                    .format(k, (al + end_point), (ar - end_point)))
        else:
            Zi.extend([lk, rk])
            # print("For node {} in case VI, we get 2 z points: {} and {}.".format(k, lk, rk))
            logger.info("For node {} in case VI, we get 2 z points: {} and {}.".format(k, lk, rk))

        zi = np.sort(np.unique(np.array(list(Zi)))).tolist()
        Z.extend(zi)

    # print('The length of Z is {}.'.format(len(Z)))
    logger.info('The length of Z is {}.'.format(len(Z)))

    return Z


def Ea(H, B_max, Beta_max, delta, a):
    sum_I, L_I, L_II = divide_set(a, B_max, Beta_max, H)
    Beta = Beta_max.copy()
    B = Beta_max / (a * H)
    if sum_I < 1:
        L_IIi, L_IIii = [], []
        X = []

        sum_bh = 0
        for k in L_II:
            X.append(Beta_max[k] - a * H[k] * B_max[k])
            sum_bh = sum_bh + B_max[k] * H[k]
        if len(X) == 0:
            # print("The sum_I is {} when a={}.".format(sum_I, a))
            logger.info("The sum_I is {} when a={}.".format(sum_I, a))
        xmax = max(X)
        # the computation of a_th in (28).
        a_star = (1 - sum_I) / sum_bh

        if a <= a_star:
            x = Fa_bi(0, 2 * xmax, max_for_equation, tol_for_equation, Beta_max, H, B_max, a)
            if x == -10000 or x == -20000:
                # print("Root Error in func Ea when a={}.".format(a))
                logger.info("Root Error in func Ea when a={}.".format(a))
                result = a * a * delta * delta
            else:
                for k in L_II:
                    B[k] = B_max[k].copy()
                    if x >= (Beta_max[k] - a * H[k] * B_max[k]):
                        L_IIi.append(k)
                        Beta[k] = Beta_max[k].copy()
                    else:
                        L_IIii.append(k)
                        Beta[k] = a * H[k] * B_max[k] + x

                y1 = 0
                for k in L_IIi:
                    y1 = y1 + math.pow(a * H[k] * B_max[k] - Beta_max[k], 2)

                y2 = len(L_IIii) * math.pow(x, 2)

                result = y1 + y2 + a * a * delta * delta
                # print('The length of L_I, L_IIi and L_IIii are {}, {} and {}.'
                #       .format(len(L_I), len(L_IIi), len(L_IIii)))
                logger.info('The length of L_I, L_IIi and L_IIii are {}, {} and {}.'
                            .format(len(L_I), len(L_IIi), len(L_IIii)))
        else:
            result = a * a * delta * delta
    else:
        result = a * a * delta * delta

    return result, Beta, B


def golden_search_Ea(amin, ama, tol, Beta_max, H, B_max, delta):
    i = 0
    while True:
        i += 1
        a1 = ama - 0.618 * (ama - amin)
        a2 = amin + 0.618 * (ama - amin)
        if -1 * Ea(H, B_max, Beta_max, delta, a1)[0] > -1 * Ea(H, B_max, Beta_max, delta, a2)[0]:
            ama = a2
        elif -1 * Ea(H, B_max, Beta_max, delta, a1)[0] <= -1 * Ea(H, B_max, Beta_max, delta, a2)[0]:
            amin = a1
        Dx = abs(ama - amin)
        if Dx <= tol:
            result_e = Ea(H, B_max, Beta_max, delta, (amin + ama) / 2)[0]
            result_a = (amin + ama) / 2
            break
        else:
            pass

    return result_e, result_a


def min_sci_Ea(amin, amax, tol, Beta_max, H, B_max, delta):
    if amin <= amax:
        def E(x):
            return Ea(H, B_max, Beta_max, delta, x)[0]

        res = minimize_scalar(E, bounds=(amin, amax), method='bounded',
                              options={'xatol': tol, 'maxiter': max_for_equation, 'disp': 0})
        return res.fun, res.x
    else:
        logger.info("The interval isn't apporiate.")
        # print("The interval isn't apporiate.")
        return None


# result -- the minimum of E(a); a_min -- the corresponding value of a.
def min_Ea(H, B_max, Beta_max, S, delta):
    Z = gen_z(H, B_max, Beta_max, S)
    Y = []
    A = []
    for i in range(len(Z) - 1):
        l = Z[i]
        r = Z[i + 1]
        if abs(l - r) <= end_point:
            y = min(Ea(H, B_max, Beta_max, delta, l)[0], Ea(H, B_max, Beta_max, delta, r)[0])
            if y == Ea(H, B_max, Beta_max, delta, l)[0]:
                A.append(l)
            else:
                A.append(r)
            Y.append(y)
        else:
            re, ra = golden_search_Ea(l, r, tol_for_min, Beta_max, H, B_max, delta)
            Y.append(re)
            A.append(ra)
    result = min(Y)
    a_min = A[Y.index(result)]

    return result, a_min


# Attention, the amax we get is little bigger than the true a_max defined in P19!
# max_index the corresponding node index
def amax(Beta_max, S):
    sum_I = 0
    i = 0
    Sn = np.sort(S)
    # print("The arranged S is {}.".format(Sn))
    Si = np.argsort(S)
    while sum_I < 1:
        if i < len(S):
            sum_I = sum_I + Beta_max[Si[i]]
            # print("The Beta value is {}, the sum_I is {}, the node is {}.".format(Beta_max[Si[i]], sum_I, Si[i]))
            logger.info("The Beta value is {}, the sum_I is {}, the node is {}.".format(Beta_max[Si[i]], sum_I, Si[i]))
        else:
            # print("amax is equal S_K")
            logger.info("amax is equal S_K")
            break
        i = i + 1
    amax = Sn[i - 1] * 1.01  # 为了避免计算机固有误差带来的困扰, 因为S的计算涉及分式
    max_index = Si[i - 1]

    return amax, max_index


# 变量aT应是一个与节点有关的变量值, 所以此处考虑输入数组。
def cop_Ea(a, AT, bmax, Beta, delta, H):
    B = []
    for i in range(len(H)):
        aT = AT[i]
        if a > aT:
            b = Beta[i] / (a * H[i])
        else:
            b = bmax
        B.append(b)
    B = np.array(B)
    # C = np.power(a * H * B - Beta, 2)
    # s = np.sum(C, axis=0) + a ** 2 * delta ** 2
    s = np.dot(a * H * B - Beta, a * H * B - Beta) + a * a * delta * delta

    return s, B


def cop(bmax, H, D, delta):
    Beta = D / np.sum(D, axis=0)
    AT = Beta / (bmax * H)
    AU = np.unique(AT)
    An = np.sort(AU)

    Ea = []
    A = []
    for i in range(len(An) + 1):
        if i == 0:
            al = end_point
            ar = An[i]
        elif i != len(An):
            al = An[i - 1] + end_point
            ar = An[i]
        else:
            al = An[i - 1] + end_point
            ar = 10 ** 18

        B = cop_Ea(al, AT, bmax, Beta, delta, H)[1]
        # Cl = (al * H * B - Beta) * H * B
        # El_prime = 2 * np.sum(Cl, axis=0) + 2 * al * delta ** 2
        El_prime = 2 * np.dot(al * H * B - Beta, H * B) + 2 * al * delta ** 2
        # Cr = (ar * H * B - Beta) * H * B
        # Er_prime = 2 * np.sum(Cr, axis=0) + 2 * ar * delta ** 2
        Er_prime = 2 * np.dot(ar * H * B - Beta, H * B) + 2 * ar * delta ** 2
        if El_prime * Er_prime < 0:
            a_star = (np.sum(Beta * H * B, axis=0)) / (np.dot(np.power(H, 2), np.power(B, 2)) + delta ** 2)
        elif El_prime >= 0:
            a_star = al
        elif Er_prime <= 0:
            a_star = ar
        s = cop_Ea(a_star, AT, bmax, Beta, delta, H)[0]
        Ea.append(s)
        A.append(a_star)

    Y = min(Ea)
    a = A[Ea.index(Y)]

    return Y, a, B


def TPC(bmax, H, delta):
    n = len(H)
    Ph = np.sort(bmax * bmax * H * H)
    Bh = np.sort(bmax * H)
    idx = np.argsort(bmax * bmax * H * H)
    Ph2 = np.cumsum(Ph)
    Ph1 = np.cumsum(Bh)
    eta_over = np.power((np.ones(n, dtype=float) * delta * delta + Ph2) / Ph1, 2)
    eta_star = np.zeros(n, dtype=float)
    for i in range(n):
        if i == n - 1:
            eta_star[i] = min(1000000, max(eta_over[i], Ph[i]))
        else:
            eta_star[i] = min(Ph[i + 1], max(eta_over[i], Ph[i]))
    F_eta = np.zeros(n, dtype=float)
    for i in range(n):
        n2 = np.power(np.sort(bmax * H) * 1 / math.sqrt(eta_star[i]) - np.ones(n, dtype=float), 2)
        F_eta[i] = delta * delta / eta_star[i] + np.cumsum(n2)[i]
    i_star = np.argmin(F_eta)
    # k_star = idx[i_star]
    eta_opt = eta_over[i_star]
    b = np.zeros(n, dtype=float)
    for i in range(n):
        if i <= i_star:
            b[idx[i]] = bmax
        else:
            b[idx[i]] = math.sqrt(eta_opt) / H[idx[i]]
    n2 = np.power(b * H * 1 / math.sqrt(eta_opt) - np.ones(n, dtype=float), 2)
    Ea = delta ** 2 / (eta_opt * n * n) + 1 / (n * n) * np.sum(n2)

    return eta_opt, b, Ea

