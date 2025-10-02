
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Impacto de la Falta de Integración", layout="wide")
st.title("Impacto de la Falta de Integración")
st.caption("Calcula el efecto en costes, tiempos de entrega y experiencia del cliente (OTIF, penalizaciones, devoluciones).")

# ---------- Parámetros base ----------
st.sidebar.header("Parámetros (base diaria)")
DAYS = st.sidebar.slider("Horizonte (días)", 15, 120, 45, step=5)
price_per_unit = st.sidebar.number_input("Precio medio por unidad (€)", 5.0, 500.0, 25.0, step=1.0)
base_demand = st.sidebar.number_input("Demanda media (unid/día)", 100, 20000, 1200, step=50)
demand_cv = st.sidebar.slider("Variabilidad demanda (CV %)", 0, 100, 25, step=5)

prod_cap = st.sidebar.number_input("Capacidad de producción (unid/día)", 100, 50000, 1400, step=100)
prod_eff = st.sidebar.slider("Eficiencia producción (%)", 40, 100, 85, step=5)

wh_capacity = st.sidebar.number_input("Capacidad almacén (unid)", 1000, 200000, 30000, step=1000)
holding_cost = st.sidebar.number_input("Coste de almacenamiento (€/unid/día)", 0.0, 5.0, 0.05, step=0.01)
stockout_penalty = st.sidebar.number_input("Penalización por rotura (€/unid no servida)", 0.0, 100.0, 10.0, step=1.0)

fleet_cap = st.sidebar.number_input("Capacidad transporte (unid/día)", 100, 50000, 1300, step=100)
transport_cost_unit = st.sidebar.number_input("Coste transporte estándar (€/unid)", 0.1, 100.0, 1.2, step=0.1)
corrective_cost_unit = st.sidebar.number_input("Coste transporte correctivo (€/unid)", 0.1, 200.0, 3.5, step=0.1)

sla_days = st.sidebar.slider("SLA de entrega (días)", 0, 15, 3, step=1)

# ---------- Parámetros de integración ----------
st.sidebar.header("Efectos de la integración (mejoras)")
forecast_improvement = st.sidebar.slider("Mejora de forecast (↓CV puntos)", 0, 40, 15, step=5)
sync_bonus = st.sidebar.slider("Mejor sincronización prod-log (↑capacidad efectiva %)", 0, 30, 10, step=5)
route_planning_gain = st.sidebar.slider("Optimización rutas (↓coste transp. %)", 0, 40, 15, step=5)
last_mile_gain = st.sidebar.slider("Mejor última milla (↑capacidad entrega %)", 0, 30, 10, step=5)

np.random.seed(10)

def simulate(integrated: bool):
    # Demanda estocástica
    mu = base_demand
    cv = max(0.0, (demand_cv - (forecast_improvement if integrated else 0)) / 100.0)
    sigma = mu * cv
    demand = np.maximum(0, np.random.normal(mu, sigma, DAYS).astype(int))

    # Capacidad efectiva
    prod_capacity_eff = int(prod_cap * (prod_eff/100.0) * (1 + (sync_bonus/100.0 if integrated else 0)))
    fleet_capacity_eff = int(fleet_cap * (1 + (last_mile_gain/100.0 if integrated else 0)))

    # Coste transporte unitario
    transport_unit_eff = transport_cost_unit * (1 - (route_planning_gain/100.0 if integrated else 0))

    # Estados
    inv = 0
    backlog = 0
    delivered = np.zeros(DAYS, dtype=int)
    otif_flags = np.zeros(DAYS, dtype=bool)
    holding_costs = 0.0
    corrective_costs = 0.0
    penalties = 0.0

    # Simulación simple día a día
    for t in range(DAYS):
        # Producción del día
        produced = prod_capacity_eff

        # Inventario entra
        inv += produced

        # Satisfacer demanda (limitado por inventario y flota)
        request = demand[t] + backlog
        shipable = min(inv, fleet_capacity_eff, request)
        delivered[t] = shipable

        # Calcular OTIF: si shipable == demand[t] (no backlog adicional) y shipable <= fleet, consideramos a tiempo
        served_today = min(demand[t], shipable)
        on_time = served_today == demand[t] and backlog == 0 and shipable <= fleet_capacity_eff
        otif_flags[t] = on_time

        # Actualizar inventario y backlog
        inv -= shipable
        backlog = max(0, request - shipable)

        # Costes de almacenamiento
        holding_costs += min(inv, wh_capacity) * holding_cost
        # Si inventario excede capacidad, asumimos sobrecoste implícito vía penalización de espacio perdido (simplemente almacenamos igual)

        # Costes por transporte
        standard_units = min(shipable, fleet_capacity_eff)
        corrective_units = max(0, shipable - fleet_capacity_eff)
        # En esta simplificación, si shipable > fleet, el exceso se marca como correctivo
        corrective_costs += corrective_units * (corrective_cost_unit - transport_unit_eff) if corrective_units>0 else 0

        # Penalización por rotura (no servido hoy)
        not_served = max(0, demand[t] - served_today)
        penalties += not_served * stockout_penalty

    # Métricas
    revenue = demand.sum() * price_per_unit  # intención de venta
    sales_real = delivered.sum() * price_per_unit  # ventas reales (entregado)
    transport_costs = delivered.sum() * transport_unit_eff + corrective_costs
    total_costs = holding_costs + transport_costs + penalties
    cost_over_sales = (total_costs / max(sales_real, 1)) * 100.0

    avg_delivery = np.nan  # simplificación de tiempos; con integración asumimos mejor capacidad => mayor OTIF
    otif = (otif_flags.sum() / DAYS) * 100.0

    return {
        "demand_series": demand,
        "delivered_series": delivered,
        "revenue": revenue,
        "sales_real": sales_real,
        "holding_costs": holding_costs,
        "transport_costs": transport_costs,
        "corrective_costs": corrective_costs,
        "penalties": penalties,
        "total_costs": total_costs,
        "cost_over_sales": cost_over_sales,
        "otif": otif
    }

no_int = simulate(False)
yes_int = simulate(True)

# ---------- KPIs comparativos ----------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("OTIF sin integración", f"{no_int['otif']:.1f}%")
c2.metric("OTIF con integración", f"{yes_int['otif']:.1f}%")
c3.metric("Coste/Sales sin int.", f"{no_int['cost_over_sales']:.1f}%")
c4.metric("Coste/Sales con int.", f"{yes_int['cost_over_sales']:.1f}%")
delta_sales = (yes_int['sales_real'] - no_int['sales_real'])/max(no_int['sales_real'],1)*100
c5.metric("Δ Ventas reales", f"{delta_sales:.1f}%")

# ---------- Series ----------
st.subheader("Demanda vs Entregado")
df = pd.DataFrame({
    "día": np.arange(1, len(no_int["demand_series"])+1),
    "Demanda": no_int["demand_series"],
    "Entregado (sin int.)": no_int["delivered_series"],
    "Entregado (con int.)": yes_int["delivered_series"]
})
st.line_chart(df.set_index("día"))

# ---------- Costes ----------
st.subheader("Desglose de costes")
costs_df = pd.DataFrame({
    "Escenario": ["Sin integración","Con integración"],
    "Almacenamiento": [no_int["holding_costs"], yes_int["holding_costs"]],
    "Transporte": [no_int["transport_costs"], yes_int["transport_costs"]],
    "Correctivo": [no_int["corrective_costs"], yes_int["corrective_costs"]],
    "Penalizaciones (roturas)": [no_int["penalties"], yes_int["penalties"]],
})
st.bar_chart(costs_df.set_index("Escenario"))

# ---------- Resumen ----------
st.subheader("Resumen")
st.write(pd.DataFrame({
    "Escenario": ["Sin integración","Con integración"],
    "Ventas reales (€)": [no_int["sales_real"], yes_int["sales_real"]],
    "Costes totales (€)": [no_int["total_costs"], yes_int["total_costs"]],
    "Coste / ventas (%)": [no_int["cost_over_sales"], yes_int["cost_over_sales"]],
    "OTIF (%)": [no_int["otif"], yes_int["otif"]],
}).set_index("Escenario").round(2))

st.caption("Ajusta los controles laterales para ver cómo la integración (mejor forecast, sincronización y rutas) mejora OTIF, reduce costes y penalizaciones.")
