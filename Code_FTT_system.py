import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy
import math
import functools
from IPython.display import clear_output
from qutip import *
from functools import reduce
from operator import mul
from collections import deque


def embed(op, which, dims):
    mats = [np.eye(d, dtype = complex) for d in dims]
    mats[which] = op
    return reduce(np.kron, mats)


class Qubit:
    
    def __init__(self, name, qubit_type, levels, EC, EJ, EL = None, phi_ext = None):
        self.name = name
        self.qubit_type = qubit_type
        self.levels = levels
        self.EC = EC
        self.EJ = EJ
        self.EL = EL if EL else 0.0
        self.phi_ext = phi_ext if phi_ext else 0.0

    def spectrum(self):
        vals, T = np.zeros((self.levels)), np.zeros((self.levels, self.levels), dtype = complex)
        
        if self.qubit_type == 'Transmon': 
            E_J = self.EJ[0]
            w01 = np.sqrt(8*self.EC*E_J)-self.EC
            for n in range(self.levels):
                vals[n] = n*w01 - 0.5*self.EC*n*(n-1)
            g0 = 1/np.sqrt(2*np.sqrt(8*self.EC/E_J))
            for n in range(self.levels-1):
                amp = np.sqrt(n+1)*g0
                T[n, n+1] = 1j*amp
                T[n+1, n] = -1j*amp
            return vals, T
        
        if self.qubit_type == 'Fluxonium':
            E_J = self.EJ[0]
            n_op = 1j/np.sqrt(2) * (self.EL/(8*self.EC))**0.25 * (create(100) - destroy(100))
            phi_op =  1/np.sqrt(2) * ((8*self.EC)/self.EL)**0.25 * (create(100) + destroy(100))
            H = (4*self.EC*n_op**2 + E_J*(identity(100) - phi_op.cosm()) + 0.5*self.EL*(phi_op - self.phi_ext*identity(100))**2)
            H_mat = np.real_if_close(H.full(), tol = 1)
            vals, vecs = scipy.sparse.linalg.eigsh(H_mat, k = min(self.levels+4, 99), sigma=0.0)
            idx = np.argsort(vals)
            vals, vecs = vals[idx], vecs[:, idx]
            vals -= vals[0]
            vals, vecs = vals[:self.levels], vecs[:, :self.levels]
            n_mat = n_op.full()
            for i in range(self.levels):
                for j in range(self.levels):
                    Tij = np.dot(vecs[:, i], np.dot(n_mat, vecs[:, j]))
                    if i > j:
                        T[j, i] = 1j*np.abs(Tij)
                    elif i < j:
                        T[j, i] = -1j*np.abs(Tij)
                    else:
                        T[j, i] = 0.0
            return vals, T
        
        if self.qubit_type == 'Tunable_Transmon':
            d = np.abs(self.EJ[0]-self.EJ[1])/(self.EJ[0]+self.EJ[1])
            EJ_eff = (self.EJ[0]+self.EJ[1])*np.sqrt(np.cos(self.phi_ext)**2 + d**2*np.sin(self.phi_ext)**2)
            w01 = np.sqrt(8*self.EC*EJ_eff)-self.EC
            for n in range(self.levels):
                vals[n] = n*w01 - 0.5*self.EC*n*(n-1)
            g0 = 1/np.sqrt(2*np.sqrt(8*self.EC/EJ_eff))
            for n in range(self.levels-1):
                amp = np.sqrt(n+1)*g0
                T[n, n+1] = 1j*amp
                T[n+1, n] = -1j*amp
            return vals, T


class System:
    
    def __init__(self, C_matrix, quantbits = None, EC = None):
        self.C_matrix = C_matrix
        self.quantbits = quantbits if quantbits else []
        self.level_arr = [qb.levels for qb in self.quantbits]
        self.dim = self._calc_dim()
        self.EC = EC if EC else np.zeros_like(C_matrix)
        
    def _calc_dim(self):
        if not self.level_arr:
            return 1
        return reduce(mul, self.level_arr, 1)

    def add_qubit(self, qubit):
        self.quantbits.append(qubit)
        self.level_arr = [qb.levels for qb in self.quantbits]
        self.dim = self._calc_dim()

    def get_EC_matrix(self, T = None, red = None):
        E_C = np.linalg.inv(self.C_matrix)*scipy.constants.e**2/(2*scipy.constants.h)
        if T is not None:
            E_C = (np.linalg.inv(T) @ np.linalg.inv(self.C_matrix) @ np.linalg.inv(T))*scipy.constants.e**2/(2*scipy.constants.h)
        if red is not None: 
            # get rid of modes that decouple
            E_C = np.asarray(E_C)
            E_C = np.delete(E_C, red, axis = 0)  
            E_C = np.delete(E_C, red, axis = 1) 
        if self.quantbits:
            for i in range(len(self.quantbits)):
                # the real EC values for the qubits are derived from the inverse elements of the capacitance -> dependent on coupling capacitances and need therefore to be updated here
                self.quantbits[i].EC = E_C[i, i]
        self.EC = E_C
        return(E_C)
        
    def get_H(self):
        if not self.quantbits:
            return Qobj(np.zeros((1,1), dtype=complex))       
        K = self.EC
        dims = self.level_arr
        dim = self.dim
        H = np.zeros((dim, dim), dtype=complex)
        local_E = []
        local_T = []
        for i, qb in enumerate(self.quantbits):
            vals, T = qb.spectrum()
            local_E.append(vals)
            local_T.append(T)
            H += embed(np.diag(vals), i, dims)      
        for i in range(len(self.quantbits)):
            for j in range(i+1, len(self.quantbits)):
                Ni = local_T[i]
                Nj = local_T[j]
                H += (K[i, j]+K[j, i]) * (embed(Ni, i, dims) @ embed(Nj, j, dims))
                
        H = Qobj(2*np.pi*H, dims = [dims, dims])
        return(0.5*(H + H.dag()))
    
    def get_eigenstate(self, state):
        state = state.unit()
        H = self.get_H()
        vals, vecs = np.linalg.eigh(H.full())
        overlaps = np.abs(vecs.conj().T @ state.full())**2
        pos_eig = np.argmax(overlaps)
        dims = [self.level_arr, [1]*len(self.level_arr)]
        ev = Qobj(vecs[:, pos_eig], dims=dims)
        return ev

    def get_energy(self, state):
        state = state.unit()
        H = self.get_H()
        E = np.vdot(state.full(), H.full() @ state.full()).real
        return float(E)
    
    def get_ZZ(self, Target):
        exc, E = np.zeros((4, len(self.level_arr)), dtype = int), []
        exc[1, Target[0]], exc[2, Target[1]], exc[3, Target[0]], exc[3, Target[1]] = 1, 1, 1, 1
        for i in range(4):
            E.append(self.get_energy(self.get_eigenstate(reduce(tensor, [basis(self.level_arr[j], exc[i, j]) for j in range(len(self.level_arr))]))))
        return(E[3]-E[2]-E[1]+E[0])


class H_drive:
    
    def __init__(self, operator, pulse_params, args = None):
        self.operator = operator
        self.pulse_params = pulse_params
        self.args = args if args else {}

    def get_envelope(self, t): 
        if self.pulse_params.get('env', 'rectangular') == 'rectangular':
            if self.pulse_params.get('t_start', 0.0) <= t and self.pulse_params.get('t_final', 0.0) >= t:
                env = self.pulse_params.get('Amp', 0.0)
            else : 
                env = 0.0
        if self.pulse_params.get('env', 'rectangular') == 'gaussian_flattop':
            if self.pulse_params.get('t_start', 0.0) <= t and self.pulse_params.get('t_final', 0.0) >= t:
                env = self.pulse_params.get('Amp', 0.0)*0.25*(1 + math.erf((t-self.pulse_params.get('t_start', 0.0)-self.pulse_params.get('tau', 1e-15))/self.pulse_params.get('tau', 1e-15)))*(1 + math.erf((self.pulse_params.get('t_final', 0.0)-t-self.pulse_params.get('tau', 1e-15))/self.pulse_params.get('tau', 1e-15)))
            else : 
                env = 0.0
        else : 
            env = 0.0
        return(env)
    
    def multitone_pulse(self, t):
        params = self.pulse_params
        pulse = 0
        for i in range(len(self.pulse_params['multi_pulses'])):
            self.pulse_params = params['multi_pulses'][i]
            pulse += self.get_envelope(t)*np.cos(self.pulse_params.get('omega_d')*t + self.pulse_params.get('phi', 0))
        self.pulse_params = params
        return(pulse)
        
    def get_pulse(self, t, args = None):
        pulse = 0
        if self.pulse_params['pulse'] == 'AC':
            env = self.get_envelope(t)
            pulse = env*np.cos(self.pulse_params.get('omega_d')*t + self.pulse_params.get('phi', 0))        
        if self.pulse_params['pulse'] == 'DC':
            env = self.get_envelope(t)
            pulse == env
        if self.pulse_params['pulse'] == 'AC_special':
            #implemented only for phi_DC = pi/2
            phi_ext = self.multitone_pulse(t)
            d = np.abs(self.args['EJ'][0]-self.args['EJ'][1])/(self.args['EJ'][0]+self.args['EJ'][1])
            drive = np.sqrt(np.cos(np.pi/2+phi_ext)**2 + d**2*np.sin(np.pi/2+phi_ext)**2)
            pulse = 2*np.pi*((self.args['EJ'][0]+self.args['EJ'][1])*drive - np.abs(self.args['EJ'][0]-self.args['EJ'][1]))
        else : 
            pulse = self.get_envelope(t)
        return(pulse)
      
    def get_t_final(self):
        if 't_final' in self.pulse_params:
            return self.pulse_params.get('t_final')
        if 'multi_pulses' in self.pulse_params:
            return(max([self.pulse_params['multi_pulses'][i].get('t_final', 0.0) for i in range(len(self.pulse_params['multi_pulses']))]))
    
    def plot_pulse(self):
        t, pulse = np.linspace(0, self.get_t_final(), 5000), np.zeros(5000)
        for i in range(5000):
            pulse[i] = self.get_pulse(t[i])
        fig, ax = plt.subplots(figsize = (10, 5))
        plt.plot(t*1e9, pulse)
        plt.xlabel('Time in ns')
        plt.ylabel('Amplitude')
        plt.show()


class TwoQ_Gate: 
    
    def __init__(self, Target_U, Target_Q, system, H_drives = None, U = None):
        self.Target_U = Target_U
        self.Target_Q = Target_Q
        self.system = system
        self.H_drives = H_drives if H_drives else []
        self.U = U if U is not None else np.zeros_like(Target_U, dtype = complex)
        
    def add_H_drive(self, drive):
        self.H_drives.append(drive)
        
    def get_t_gate(self):
        return max((d.get_t_final() for d in self.H_drives), default = 0.0)
    
    def simulate_state(self, state):
        H = [self.system.get_H()]
        tau = np.linspace(0, self.get_t_gate(), 5000)
        for drive in self.H_drives:
            H.append([drive.operator, drive.get_pulse])
        result = sesolve(H, state, tau, options = {'nsteps' : 20000, 'atol' : 1e-15, 'rtol' : 1e-15}) 
        return(result.states[-1])
    
    def simulate_TE_state(self, state):
        H = [self.system.get_H()]
        tau = np.linspace(0, self.get_t_gate(), 5000)
        for drive in self.H_drives:
            H.append([drive.operator, drive.get_pulse])
        result = sesolve(H, state, tau)
        return(result.states)

    def get_fidelity(self):
        return(np.abs(np.trace(self.Target_U @ np.conj(self.U).T))/self.Target_U.shape[0])
    
    def extract_unitary(self):
        def Z_correction(phi):
            U_phase = np.diag([np.exp(1j*(-phi[0]-phi[1])), np.exp(1j*(-phi[0]+phi[1])), np.exp(1j*(phi[0]-phi[1])), np.exp(1j*(phi[0]+phi[1]))])
            return(1-np.abs(np.trace(self.Target_U @ np.conj(self.U @ U_phase).T))/self.Target_U.shape[0])

        dims = self.system.level_arr
        exc = np.zeros((4, len(dims)), dtype = int)
        exc[1, self.Target_Q[0]], exc[2, self.Target_Q[1]], exc[3, self.Target_Q[0]], exc[3, self.Target_Q[1]] = 1, 1, 1, 1

        states_i, states_f = [], []
        for i in range(4):
            states_i.append(self.system.get_eigenstate(reduce(tensor, [basis(dims[j], exc[i, j]) for j in range(len(dims))])))
            states_f.append(self.simulate_state(states_i[i]))
        
        self.U = np.zeros((4, 4), dtype = complex)
        for i in range(4):
            for j in range(4):
                self.U[i, j] = (states_i[i].dag().full() @ states_f[j].full())[0, 0]
        phi_opt = opt.minimize(Z_correction, [3*np.pi/2, np.pi/2], bounds = [[0, 2*np.pi], [0, 2*np.pi]], method = 'Nelder-mead').x
        U_phase = np.diag([np.exp(1j*(-phi_opt[0]-phi_opt[1])), np.exp(1j*(-phi_opt[0]+phi_opt[1])), np.exp(1j*(phi_opt[0]-phi_opt[1])), np.exp(1j*(phi_opt[0]+phi_opt[1]))])
        self.U = self.U @ U_phase
        return(self.U)
    
    def optimize_fidelity(self, init_params, param_map, bounds, method = 'Nelder-Mead'):
                    
        def cost_function(x):
            for val, (d_idx, p_name) in zip(x, param_map):
                if 'multi_pulses' in self.H_drives[d_idx].pulse_params:
                    self.H_drives[d_idx].pulse_params['multi_pulses'][p_name[0]][p_name[1]] = val
                else : 
                    self.H_drives[d_idx].pulse_params[p_name] = val
            self.U = self.extract_unitary()
            f = self.get_fidelity()
            val = 1 - f 
            print(val)
            print(x)
            return(val)

        result = opt.minimize(cost_function, x0 = init_params, bounds = bounds, method = method)
        for val, (d_idx, p_name) in zip(result.x, param_map):
            if 'multi_pulses' in self.H_drives[d_idx].pulse_params:
                self.H_drives[d_idx].pulse_params['multi_pulses'][p_name[0]][p_name[1]] = val
            else : 
                self.H_drives[d_idx].pulse_params[p_name] = val

        print("Optimized parameters:")
        for (d_idx, p_name), val in zip(param_map, result.x):
            print(f"Drive {d_idx}, {p_name} = {val}")
        print("Final infidelity:", result.fun)
        return result




# FTT-System with parameters of table 1 in https://www.arxiv.org/pdf/2508.09267

levels = [5, 5, 5] # number of levels simulated for each qubit/coupler

C = np.array([[24e-15, -6e-15, 0, 0],
              [-6e-15, 66e-15, -22e-15, 0], 
              [0, -22e-15, 75.5e-15, -15.5e-15], 
              [0, 0, -15.5e-15, 103.3e-15]])

T = np.array([[1, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0.5, -0.5, 0], [0, 0, 0, 1]]) #Transformation matrix 

Q1 = Qubit(name = 'Q1', qubit_type = 'Fluxonium', levels = levels[0], EC = 0.828e9, EJ = [6.1e9], EL = 1.6e9, phi_ext = np.pi)
Cp = Qubit(name = 'Coupler', qubit_type = 'Tunable_Transmon', levels = levels[1], EC = 0.428e9, EJ = [7.5e9, 12.822e9], phi_ext = np.pi/2)
Q2 = Qubit(name = 'Q2', qubit_type = 'Transmon', levels = levels[2], EC = 0.194e9, EJ = [13.6e9])

S = System(C)
S.add_qubit(Q1)
S.add_qubit(Cp)
S.add_qubit(Q2)
EC = S.get_EC_matrix(T = T, red = [1])

print(S.get_ZZ([0, 2])/(2e3*np.pi), 'kHz')


# 40ns CZ-gate

t_gate = 40e-9

phi_ZPF = (2*Cp.EC/np.abs(Cp.EJ[0]-Cp.EJ[1]))**0.25
b_plus_bdag = tensor(identity(levels[0]), create(levels[1]) + destroy(levels[1]), identity(levels[2]))
drive_op = phi_ZPF**2*b_plus_bdag**2/2
drive_op += - phi_ZPF**4*b_plus_bdag**4/24


# Example simulating two-tone drives

params1 = {'Amp' : 0.2111687711, 'omega_d' : 1075918692, 'phi' : 1.9771, 'env' : 'gaussian_flattop', 'tau' : 3.812439357e-9, 't_final' : 40.117435934e-9} # first drive tone
params2 = {'Amp' : 0.0015711658, 'omega_d' : 1740729227, 'phi' : 1.2076, 'env' : 'gaussian_flattop', 'tau' : 4.028123557e-9, 't_final' : 33.834262193e-9, 't_start' : 6.283173741} # second drive tone
params = {'multi_pulses' : [params1, params2], 'pulse' : 'AC_special', 't_gate' : t_gate}
args = {'EJ' : Cp.EJ}
Drive = H_drive(drive_op, params, args = args) # Drive Hamiltonian + parameters

U_CZ = np.diag([1, 1, 1, -1]) # ideal unitary
CZ = TwoQ_Gate(U_CZ, [0, 2], S) # initialize 2Q-Gate (ideal_U, Target: CZ should operate on Q1 and Q2, System)
CZ.add_H_drive(Drive) # add drives

#CZ.extract_unitary()
#print(CZ.get_fidelity())


# Example optimizing 40ns CZ-gate

params1 = {'Amp' : 0.211, 'omega_d' : 1076000000, 'phi' : 0.473, 'env' : 'gaussian_flattop', 'tau' : 3.77e-9, 't_final' : 40e-9}
params = {'multi_pulses' : [params1], 'pulse' : 'AC_special'}
Drive = H_drive(drive_op, params, args = args)

U_CZ = np.diag([1, 1, 1, -1])
CZ = TwoQ_Gate(U_CZ, [0, 2], S)
CZ.add_H_drive(Drive)

param_map = [(0, [0, 'Amp']), (0, [0, 'omega_d']), (0, [0, 'phi']), (0, [0, 'tau'])]
bounds = [[0.205, 0.215], [1070000000, 1080000000], [0, 2*np.pi], [3e-9, 4e-9]]
init_vals = [params1['Amp'], params1['omega_d'], params1['phi'], params1['tau']]

result = CZ.optimize_fidelity(init_vals, param_map, bounds)




