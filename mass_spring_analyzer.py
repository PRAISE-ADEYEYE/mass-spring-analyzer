# ------------------ MODULE 1 ------------------
# Introduction, Imports, and Setup

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, Eq, solve, simplify
from tabulate import tabulate
import streamlit as st

# Global Constants
PI = np.pi
g = 9.81  # m/s^2

st.set_page_config(page_title="Mass-Spring System Analyzer", layout="centered")

st.title("🔧 Mass-Spring System Analysis Tool")
st.markdown("""
Solving for the unknown original mass and spring constant
when subjected to vibrating force using natural frequency.
""")

# ------------------ MODULE 2 ------------------
# Symbolic Derivation using SymPy

st.header("📐 Step 1: Symbolic Derivation Using Natural Frequency Formula")

# Define symbols
m1, k = symbols('m1 k')
f1 = 3.56  # Hz
f2 = 2.9  # Hz
delta_m = 5

# Frequency formula: f = (1 / 2pi) * sqrt(k / m)
# Rearranged: k = (2pi * f)^2 * m
k_expr1 = (2 * PI * f1) ** 2 * m1
k_expr2 = (2 * PI * f2) ** 2 * (m1 + delta_m)

# Set up the equation k1 = k2
equation = Eq(k_expr1, k_expr2)
st.latex(equation)

# Solve for m1
m1_solution = solve(equation, m1)[0]
k_solution = k_expr1.subs(m1, m1_solution)

st.success(f"Solved unknown mass: m₁ = {m1_solution:.4f} kg")
st.success(f"Solved spring constant: k = {k_solution:.4f} N/m")


# ------------------ MODULE 3 ------------------
# Class Definitions and System Modeling

def spring_constant(mass, frequency):
    return (2 * PI * frequency) ** 2 * mass


def force(mass):
    return mass * g


class MassSpringSystem:
    def __init__(self, f1, f2, delta_mass):
        self.f1 = f1
        self.f2 = f2
        self.delta_m = delta_mass
        self.m1 = None
        self.k = None

    def solve_unknowns(self):
        f1sq = self.f1 ** 2
        f2sq = self.f2 ** 2
        ratio = f1sq / f2sq
        self.m1 = self.delta_m / (ratio - 1)
        self.k = (2 * PI * self.f1) ** 2 * self.m1
        return self.m1, self.k

    def get_summary(self):
        return {
            "Natural Frequency f1 (Hz)": self.f1,
            "Natural Frequency f2 (Hz)": self.f2,
            "Added Mass (kg)": self.delta_m,
            "Calculated Mass m1 (kg)": round(self.m1, 4),
            "Spring Constant k (N/m)": round(self.k, 4)
        }


# ------------------ MODULE 4 ------------------
# Plotting Functions

def plot_mass_vs_force(masses):
    forces = masses * g
    fig, ax = plt.subplots()
    ax.plot(masses, forces, label="F = mg", color='blue')
    ax.set_title('Graph 1: Mass vs Force')
    ax.set_xlabel('Mass (kg)')
    ax.set_ylabel('Force (N)')
    ax.grid(True)
    ax.legend()
    return fig


def plot_mass_vs_spring_constant(masses, frequency):
    k_values = (2 * PI * frequency) ** 2 * masses
    fig, ax = plt.subplots()
    ax.plot(masses, k_values, label=f'k = (2πf)^2 * m (f = {frequency} Hz)', color='green')
    ax.set_title('Graph 2: Mass vs Spring Constant')
    ax.set_xlabel('Mass (kg)')
    ax.set_ylabel('Spring Constant (N/m)')
    ax.grid(True)
    ax.legend()
    return fig


def plot_frequency_vs_mass(masses):
    k_val = (2 * PI * f1) ** 2 * m1_solution
    k_val = float(k_val)
    frequencies = (1 / (2 * PI)) * np.sqrt(k_val / masses)
    fig, ax = plt.subplots()
    ax.plot(masses, frequencies, label='f = (1 / 2π) * sqrt(k/m)', color='purple')
    ax.set_title('Graph 3: Frequency vs Mass')
    ax.set_xlabel('Mass (kg)')
    ax.set_ylabel('Natural Frequency (Hz)')
    ax.grid(True)
    ax.legend()
    return fig


# ------------------ MODULE 5 ------------------
# User Interactivity

st.sidebar.header("🧠 Input Parameters")
f1_input = st.sidebar.number_input("Enter natural frequency f1 (Hz)", value=3.56)
f2_input = st.sidebar.number_input("Enter natural frequency f2 (Hz)", value=2.9)
delta_m_input = st.sidebar.number_input("Enter added mass (kg)", value=5.0)

system = MassSpringSystem(f1_input, f2_input, delta_m_input)
mass, stiffness = system.solve_unknowns()
data = system.get_summary()
st.header("📊 Results Summary")
st.table(data)

# ------------------ MODULE 6 ------------------
# Run and Plot

mass_range_1 = np.linspace(mass, mass + 20, 300)
st.pyplot(plot_mass_vs_force(mass_range_1))
st.pyplot(plot_mass_vs_spring_constant(mass_range_1, f1_input))
mass_range_2 = np.linspace(1, 20, 100)
st.pyplot(plot_frequency_vs_mass(mass_range_2))


# ------------------ MODULE 7 ------------------
# 🔄 Energy Analysis, Damped Systems, Simulations

st.header("🔋Energy Analysis")

st.subheader("Kinetic and Potential Energy")

mass_for_energy = st.number_input("Enter mass for energy analysis (kg)", value=mass)
spring_k = stiffness
velocity = st.number_input("Enter velocity (m/s)", value=2.0)
displacement = st.number_input("Enter displacement (m)", value=0.1)

kinetic_energy = 0.5 * mass_for_energy * velocity ** 2
potential_energy = 0.5 * spring_k * displacement ** 2

st.success(f"Kinetic Energy (KE) = {kinetic_energy:.2f} Joules")
st.success(f"Potential Energy (PE) = {potential_energy:.2f} Joules")

# Plot KE and PE
energy_labels = ['Kinetic Energy', 'Potential Energy']
energy_values = [kinetic_energy, potential_energy]

fig_energy, ax = plt.subplots()
ax.bar(energy_labels, energy_values, color=['orange', 'cyan'])
ax.set_title('Energy Distribution')
ax.set_ylabel('Energy (Joules)')
st.pyplot(fig_energy)

# ------------------

st.header("🌪️ Damped Systems (Light Introduction)")

st.subheader("Underdamped Oscillation Example")

c = st.number_input("Enter damping coefficient (Ns/m)", value=1.0)
wn = np.sqrt(spring_k / mass_for_energy)  # Natural frequency
wd = np.sqrt(wn**2 - (c / (2 * mass_for_energy))**2)  # Damped natural frequency

st.success(f"Natural Frequency (undamped) ωₙ = {wn:.2f} rad/s")
st.success(f"Damped Natural Frequency ω_d = {wd:.2f} rad/s")

t = np.linspace(0, 10, 500)
x = np.exp(-c * t / (2 * mass_for_energy)) * np.cos(wd * t)

fig_damped, ax = plt.subplots()
ax.plot(t, x, label="Damped Vibration", color='red')
ax.set_title('Damped System Response')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.grid(True)
ax.legend()
st.pyplot(fig_damped)

# ------------------

st.header("🎢 Simple Simulation of Mass-Spring Motion")

st.subheader("Oscillating Motion Animation")

amplitude = displacement
omega = wn
time_sim = np.linspace(0, 10, 300)
position = amplitude * np.cos(omega * time_sim)

fig_sim, ax = plt.subplots()
ax.plot(time_sim, position, label="Simple Harmonic Motion", color='blue')
ax.set_title('Mass-Spring Oscillation')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (m)')
ax.grid(True)
ax.legend()
st.pyplot(fig_sim)

# ------------------ MODULE 8 ------------------
# Footer

st.markdown("""
---
**Thank you for using the Mass-Spring System Analyzer!**  
Developed by Praise Adeyeye with ❤️ using Python, SymPy, Matplotlib, and Streamlit.
""")
