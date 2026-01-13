import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import os

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import seaborn as sns

from datetime import timedelta

from src.data.metrics import collect_vt_metrics_range, compute_metrics_with_moves
from src.utils.utils import codificacion_ciudades, get_city_name_from_code
from src.data.distance_utils import distance


def plot_metrics_comparison_dynamic(
    algorithms,
    city,
    metricas,
    dist_dict,
    fechas,
    metric_visualization_params=None,
    save_dir=None,
    
):
    """Compares any number of algorithms/historic datasets dynamically, 
    with optional relative gap display vs baseline 'Histórico'."""

    all_metrics = pd.DataFrame()

    # Compute metrics for each algorithm
    metrics_by_algo = {}
    for algo in algorithms:
        print('\n')
        df = compute_metrics_with_moves(
            algo["labors_df"], 
            algo["moves_df"], 
            fechas, 
            dist_dict,
            workday_hours=8,
            city=city,
            assignment_type=algo["type"],
            skip_weekends=False,
            dist_method='haversine'
        )
        metrics_by_algo[algo["name"]] = df

        df_altered = df.copy()
        df_altered['algo'] = algo['name']
        all_metrics = pd.concat([all_metrics, df_altered])

    if save_dir:
        all_metrics.to_csv(f'{save_dir}/results_summary.csv', index=True)

    if isinstance(metricas, str):
        metricas = [metricas]

    # Prepare x-axis
    x_vals = pd.to_datetime(metrics_by_algo[algorithms[0]["name"]]["day"]).dt.strftime("%Y-%m-%d")

    # Create one figure per metric
    for metrica in metricas:
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.35, 0.65],
            subplot_titles=["Totales", "Serie diaria"]
        )

        totals = {
            algo["name"]: metrics_by_algo[algo["name"]][metrica].sum()
            for algo in algorithms
        }

        # Get baseline (Histórico)
        baseline_val = totals.get("Histórico", None)

        # --- Bar plot (left) ---
        for algo in algorithms:
            if algo.get("visible", True):
                val = totals[algo["name"]]

                # Compute percentage or absolute gap vs baseline
                # if (
                #     baseline_val is not None 
                #     and baseline_val != 0 
                #     and metrica in metrics_with_gap
                #     and algo["name"] != "Histórico"
                # ):
                #     abs_gap = val - baseline_val
                #     perc_gap = (abs_gap / baseline_val) * 100
                #     text = f"{val:.0f}<br>{perc_gap:+.1f}%"
                # else:
                #     text = f"{val:.0f}"

                # --- NEW flexible display: metric_visualization_params ---
                show_value, show_gap = metric_visualization_params.get(metrica, (True, False))

                text_parts = []

                # Value
                if show_value:
                    text_parts.append(f"{val:.0f}")

                # Gap
                if (
                    show_gap
                    and baseline_val is not None
                    and baseline_val != 0
                    and algo["name"] != "Histórico"
                ):
                    abs_gap = val - baseline_val
                    perc_gap = (abs_gap / baseline_val) * 100
                    text_parts.append(f"{perc_gap:+.1f}%")

                # If neither value nor gap → blank label
                text = "<br>".join(text_parts) if text_parts else ""
                textposition = "auto" if text else None

                fig.add_trace(go.Bar(
                    x=[algo["name"]],
                    y=[val],
                    name=algo["name"],
                    text=[text],
                    textposition=textposition,
                    marker=dict(
                        color=algo["color"],
                        line=dict(
                            width=1.2,
                            color="rgba(0,0,0,0.35)"   # subtle outline
                        )
                    )
                ), row=1, col=1)

        # --- Time series (right) ---
        for algo in algorithms:
            if algo.get("visible", True):
                dash_style = (
                    "solid" if algo["type"] == "historic"
                    else "dashdot" if "react" in algo["name"].lower()
                    else "dash"
                )
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=metrics_by_algo[algo["name"]][metrica],
                    mode="lines+markers",
                    name=algo["name"],
                    line=dict(color=algo["color"], dash=dash_style, width=2),
                    marker=dict(size=7)
                ), row=1, col=2)

        # --- Layout ---
        titles = {
            'service_count': 'Número de servicios',
            'vt_count': 'Número labores Vehicle Transportation',
            'num_drivers': 'Número conductores',
            'driver_extra_time': 'Tiempo extra', 
            'driver_move_distance': 'Distancia en vacío'
        }
        fig.update_layout(
            height=500, width=1100,
            title=dict(
                text=f"{titles[metrica]} — {get_city_name_from_code(city)}",
                x=0.5, xanchor="center"
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5,
                font=dict(size=11)
            ),
            plot_bgcolor="white",
            margin=dict(t=100, b=120)
        )

        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.25)",
            gridwidth=1.2,
            griddash="dash",
            zeroline=False
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.25)",
            gridwidth=1.2,
            griddash="dash",
            zeroline=False
        )

        if save_dir is not None:
            file_path = os.path.join(
                save_dir,
                f"{city}_{metrica}.html"
            )

            fig.write_html(file_path, include_plotlyjs="cdn")
        else:
            fig.show()


def add_aggregated_totals(
    metrics_df: pd.DataFrame,
    group_by: list = ["alpha", "num_iterations"]
) -> pd.DataFrame:
    """
    Add aggregated totals (summing across all cities) to a metrics DataFrame.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with columns ['alpha', 'num_iterations', 'city', ...metrics].

    Returns
    -------
    pd.DataFrame
        Original DataFrame plus extra rows where city == 'ALL',
        representing sums of all cities for each (alpha, num_iterations).
    """
    # --- 1. Aggregate by alpha + num_iterations ---
    agg_df = (
        metrics_df
        .groupby(group_by, as_index=False)
        .sum(numeric_only=True)  # sum only numeric cols
    )

    # --- 2. Mark as aggregated rows ---
    agg_df["city"] = "ALL"

    # --- 3. Merge back ---
    metrics_with_total = pd.concat([metrics_df, agg_df], ignore_index=True)

    return metrics_with_total


# def plot_gantt_labors_by_driver(
#     df: pd.DataFrame, 
#     day_str: str, 
#     driver_col: str | None = None,
#     min_row_height: int = 28, 
#     max_left_margin: int = 420,
#     tickfont_size: int = 11, 
#     return_fig: bool = True
# ):
#     """
#     Genera un diagrama de Gantt para visualizar las labores realizadas por cada conductor en un día específico.

#     La función filtra las labores cuya fecha de inicio y fin ocurren completamente dentro del día indicado,
#     asigna etiquetas a cada conductor con el conteo de labores, y ajusta automáticamente el tamaño del gráfico
#     y los márgenes para optimizar la visualización de los nombres en el eje Y.

#     Parámetros
#     ----------
#     df : pd.DataFrame
#         DataFrame que contiene los datos de labores, incluyendo fechas de inicio y fin, 
#         categoría de la labor y la columna de conductor.
#     day_str : str
#         Fecha objetivo en formato reconocible por pandas (ej. "2025-08-12").
#     driver_col : str o None, opcional
#         Nombre de la columna que identifica al conductor. Si es None, se intentará detectar automáticamente
#         buscando variantes comunes ("alfred", "driver", "conductor", etc.).
#     min_row_height : int, opcional
#         Altura mínima en píxeles por fila/conductor en el gráfico. Valor por defecto: 28.
#     max_left_margin : int, opcional
#         Margen izquierdo máximo en píxeles para mostrar las etiquetas completas. Valor por defecto: 420.
#     tickfont_size : int, opcional
#         Tamaño de fuente para las etiquetas del eje Y. Valor por defecto: 11.

#     Retorno
#     -------
#     None
#         La función no retorna un valor. Muestra el diagrama de Gantt directamente utilizando Plotly.
    
#     Notas
#     -----
#     - Solo se incluyen labores cuya fecha de inicio y fin estén completamente dentro del día especificado.
#     - Si no se encuentra una columna de conductor válida, se lanzará un KeyError.
#     - Las etiquetas del eje Y incluyen el nombre del conductor y el número de labores realizadas en el día.
#     """
#     if df is None or df.empty:
#         print("⚠️ DataFrame vacío.")
#         return

#     # Detección automática de columna conductor si no se pasa como argumento
#     if driver_col is None:
#         colmap = {c.lower(): c for c in df.columns}
#         driver_col = (colmap.get('alfred') or colmap.get("alfred's") or
#                     colmap.get('assigned_driver') or colmap.get('driver') or
#                     colmap.get('conductor') or colmap.get('alfred_id'))
#         if driver_col is None:
#             raise KeyError("No se encontró columna de conductor válida.")

#     # ⏳ Conversión a datetime y limpieza de filas con fechas inválidas
#     dfp = df.copy()
#     if 'actual_start' in dfp.columns:
#             dfp['start_time'] = dfp['actual_start']
#     if 'actual_end' in dfp.columns:
#             dfp['end_time'] = dfp['actual_end']

#     dfp['start_time'] = pd.to_datetime(dfp['start_time'], errors='coerce')
#     dfp['end_time']   = pd.to_datetime(dfp['end_time'], errors='coerce')
#     dfp = dfp.dropna(subset=['start_time','end_time'])

#     # Ventana diaria [start, end) y ajuste de zona horaria si aplica
#     start = pd.to_datetime(day_str).normalize()
#     tz = dfp['start_time'].dt.tz
#     if tz is not None and start.tzinfo is None:
#         start = start.tz_localize(tz)
#     end = start + pd.Timedelta(days=1)

#     # Filtro de labores del día con horas válidas y conductor no nulo
#     in_day = (
#         dfp['start_time'].between(start, end, inclusive='left') &
#         dfp['end_time'].between(start, end, inclusive='left') &
#         (dfp['end_time'] > dfp['start_time']) &
#         (dfp[driver_col].notna())
#     )
#     dfp = dfp.loc[in_day].copy()
#     if dfp.empty:
#         print("⚠️ No hay labores con conductor para ese día.")
#         return

#     # Etiqueta con nombre y conteo de labores
#     counts = dfp.groupby(driver_col).size()
#     dfp['driver_label'] = dfp[driver_col].map(lambda d: f"{d} ({int(counts.get(d,0))})")

#     # Columnas para tooltip en hover
#     hover_cols = [c for c in [
#         'labor_name','service_id','labor_id','labor_category',
#         'start_time','end_time'
#     ] if c in dfp.columns]

#     # Ajuste dinámico de altura y margen izquierdo
#     unique_labels = dfp['driver_label'].unique().tolist()
#     n_rows = len(unique_labels)
#     max_label_len = max(len(str(x)) for x in unique_labels)
#     left_margin = min(60 + 7 * max_label_len, max_left_margin)
#     height = max(320, int(min_row_height * (n_rows + 1)))

#     # Creación del gráfico Gantt
#     fig = px.timeline(
#         dfp.sort_values(['driver_label','start_time']),
#         x_start='start_time', x_end='end_time',
#         y='driver_label',
#         color='labor_category' if 'labor_category' in dfp.columns else None,
#         hover_data=hover_cols
#     )
#     fig.update_yaxes(autorange='reversed', type='category', automargin=True, tickfont=dict(size=tickfont_size))
#     fig.update_xaxes(tickangle=-45)
#     fig.update_layout(
#         title=f"Labores por conductor — {day_str}",
#         margin=dict(l=left_margin, r=80, t=60, b=60),
#         height=height,
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#     )

#     if return_fig:
#         return fig
#     else:
#         fig.show()

def plot_gantt_by_services(
    df_cleaned: pd.DataFrame,
    day_str: str,
    assignment_type: str,
    return_fig: bool
):
    """
    Genera un gráfico tipo Gantt por servicio,
    mostrando las labores en orden cronológico.
    """
    if assignment_type == "historic":
        start_col, end_col, alfred_col = "historic_start", "historic_end", "historic_driver"
    elif assignment_type == "algorithm":
        start_col, end_col, alfred_col = "actual_start", "actual_end", "assigned_driver"
    else:
        raise ValueError("assignment_type debe ser 'historic' o 'algorithm'")

    df_cleaned_plot = df_cleaned.dropna(subset=[start_col, end_col])
    if df_cleaned_plot.empty:
        print(f"⚠️ No hay labores con tiempos válidos para {day_str}")
        return

    df_cleaned_plot["service_id_str"] = df_cleaned_plot["service_id"].astype(str)
    df_cleaned_plot["driver_label"]   = df_cleaned_plot[alfred_col].fillna("on site").astype(str)

    fig = px.timeline(
        df_cleaned_plot.sort_values(["service_id_str", start_col]),
        x_start=start_col, x_end=end_col,
        y="service_id_str", color="labor_category",
        title=f"Detalle de Labors por Servicio ({day_str})",
        hover_data=["driver_label","labor_name"]
    )
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_layout(height=max(600, len(df_cleaned_plot["service_id_str"].unique())))

    if return_fig:
        return fig
    else:
        fig.show()


def plot_gantt_by_drivers(
    df_cleaned: pd.DataFrame,
    df_moves: pd.DataFrame,
    day_str: str,
    tiempo_gracia: int,
    assignment_type: str,
    return_fig: bool
):
    """
    Genera un gráfico tipo Gantt por conductor,
    incluyendo órdenes fallidas y llegadas tarde.
    """
    if df_moves.empty:
        print(f"⚠️ No hay tareas para el día {day_str}")
        return

    # Selección dinámica de columnas
    if assignment_type == "historic":
        start_col, end_col, alfred_col = "historic_start", "historic_end", "historic_driver"
    elif assignment_type == "algorithm":
        start_col, end_col, alfred_col = "actual_start", "actual_end", "assigned_driver"
    else:
        raise ValueError("assignment_type debe ser 'historic' o 'algorithm'")

    # --- DataFrame filtrado ---
    df_plot = df_moves[df_moves[alfred_col].notna()].copy()
    df_plot[alfred_col] = df_plot[alfred_col].astype(str)

    # --- Conteo de labors ---
    labor_counts = (
        df_plot[~df_plot['labor_category'].isin(['FREE_TIME','DRIVER_MOVE'])]
        .groupby(alfred_col).size().to_dict()
    )

    y_labels = {
        drv: f"{drv} ({labor_counts.get(drv,0)})"
        for drv in df_plot[alfred_col].unique()
    }

    # --- Plot principal ---
    fig = px.timeline(
        df_plot,
        x_start=start_col, x_end=end_col,
        y=alfred_col, color="labor_category",
        color_discrete_map={'FREE_TIME':'gray', 'DRIVER_MOVE':'lightblue'},
        hover_data={
            "labor_name": True,
            "service_id": True,
            "labor_id": True,
            "schedule_date": True,
            "duration_min": True,
            "start_point": True,
            "end_point": True
        }
    )
    cats = fig.layout.yaxis.categoryarray or list(dict.fromkeys(df_plot[alfred_col]))
    ticktext = [y_labels.get(cat, cat) for cat in cats]
    fig.update_yaxes(
        autorange="reversed",
        type="category",
        categoryorder="array",
        categoryarray=cats,
        ticktext=ticktext,
        tickvals=cats
    )
    fig.update_layout(
        height=max(600, len(cats)*30),
        title=f"Tareas de Conductores ({day_str})"
    )

    # --- Órdenes fallidas ---
    fails = df_cleaned[
        (df_cleaned['labor_category']=="VEHICLE_TRANSPORTATION") &
        (df_cleaned[alfred_col].isna())
    ]
    if not fails.empty:
        x_seg, y_seg = [], []
        for sched in fails["schedule_date"]:
            x_seg += [sched, sched, None]
            y_seg += [cats[0], cats[-1], None]
        fig.add_trace(go.Scatter(
            x=x_seg, y=y_seg, mode="lines",
            line=dict(color="red", dash="dash"),
            name="Fallido", legendgroup="Fallido"
        ))
        fig.add_trace(go.Scatter(
            x=fails["schedule_date"],
            y=[cats[0]]*len(fails),
            mode="markers",
            marker=dict(symbol="line-ns-open", color="red", size=12),
            name="Fallido", legendgroup="Fallido",
            customdata=fails[["service_id","labor_id","map_start_point"]].values,
            hovertemplate=(
                "Fallido<br>Servicio: %{customdata[0]}<br>"
                "Labor: %{customdata[1]}<br>"
                "Ubicación: %{customdata[2]}<extra></extra>"
            )
        ))

    # --- Llegadas tarde ---
    late_tasks = (
        df_cleaned[df_cleaned["labor_category"]=="VEHICLE_TRANSPORTATION"]
        .sort_values("schedule_date")
        .groupby("service_id").first()
        .reset_index()
    )
    if not late_tasks.empty:
        late_tasks["late_dead"] = late_tasks["schedule_date"] + timedelta(minutes=tiempo_gracia)
        late = late_tasks[
            late_tasks[alfred_col].notna() &
            (late_tasks[start_col] > late_tasks["late_dead"])
        ].copy()
        if not late.empty:
            late["late_minutes"] = (
                (late[start_col] - late["late_dead"]).dt.total_seconds()/60
            ).round(1)
            fig.add_trace(go.Scatter(
                x=late[start_col],
                y=late[alfred_col].astype(str),
                mode="markers",
                marker=dict(symbol="x", size=12, color="black"),
                name="Llegada Tarde", legendgroup="Llegada Tarde",
                customdata=late[["service_id","labor_id","late_minutes"]].values,
                hovertemplate=(
                    "Llegada Tarde<br>Servicio: %{customdata[0]}<br>"
                    "Labor: %{customdata[1]}<br>"
                    "Retraso: %{customdata[2]} min<extra></extra>"
                )
            ))

    if return_fig:
        return fig
    else:
        fig.show()


def plot_results(
    df_cleaned: pd.DataFrame,
    df_moves: pd.DataFrame,
    day_str: str,
    tiempo_gracia: int,
    assignment_type: str):
    """
    Genera los gráficos tipo Gantt (por conductor y por servicio) y los muestra,
    incluyendo órdenes fallidas, marcadores de llegadas tarde,
    y el conteo de labors por conductor en el eje y.
    """
    if df_moves.empty:
        print(f"⚠️ No hay tareas para el día {day_str}")
        return
    
    # Selección dinámica de columnas de tiempo
    if assignment_type == "historic":
        start_col, end_col, alfred_col = "historic_start", "historic_end", "historic_driver"
    elif assignment_type == "algorithm":
        start_col, end_col, alfred_col = "actual_start", "actual_end", "assigned_driver"
    else:
        raise ValueError("assignment_type debe ser 'historic' o 'algorithm'")

    # 1) Preparar DataFrame para plot
    df_plot = df_moves[df_moves[alfred_col].notna()].copy()
    df_plot[alfred_col] = df_plot[alfred_col].astype(str)

    # 2) Calcular conteo de labors (excluyendo FREE_TIME y DRIVER_MOVE)
    labor_counts = (
        df_plot[~df_plot['labor_category'].isin(['FREE_TIME','DRIVER_MOVE'])]
          .groupby(alfred_col).size()
          .to_dict()
    )

    # 3) Construir mapping de etiquetas
    y_labels = {
        drv: f"{drv} ({labor_counts.get(drv,0)})"
        for drv in df_plot[alfred_col].unique()
    }

    # 4) Gantt por conductores
    fig = px.timeline(
        df_plot,
        x_start=start_col, x_end=end_col,
        y=alfred_col, color="labor_category",
        color_discrete_map={'FREE_TIME':'gray', 'DRIVER_MOVE':'lightblue'},
        hover_data={
            "labor_name":    True,
            "service_id":    True,
            "labor_id":      True,
            "schedule_date": True,
            "duration_min":  True,
            "start_point":   True,
            "end_point":     True
        }
    )
    cats = fig.layout.yaxis.categoryarray or list(dict.fromkeys(df_plot[alfred_col]))
    ticktext = [y_labels.get(cat, cat) for cat in cats]
    fig.update_yaxes(
        autorange="reversed",
        type='category',
        categoryorder='array',
        categoryarray=cats,
        ticktext=ticktext,
        tickvals=cats
    )
    fig.update_layout(
        height=max(600, len(cats)*30),
        title=f"Tareas de Conductores ({day_str})"
    )

    # 5) Órdenes fallidas
    fails = df_cleaned[
        (df_cleaned['labor_category']=='VEHICLE_TRANSPORTATION') &
        (df_cleaned[alfred_col].isna())
    ]
    if not fails.empty:
        x_seg, y_seg = [], []
        for sched in fails['schedule_date']:
            x_seg += [sched, sched, None]
            y_seg += [cats[0], cats[-1], None]
        fig.add_trace(go.Scatter(
            x=x_seg, y=y_seg, mode='lines',
            line=dict(color='red', dash='dash'),
            name='Fallido', legendgroup='Fallido'
        ))
        fig.add_trace(go.Scatter(
            x=fails['schedule_date'],
            y=[cats[0]]*len(fails),
            mode='markers',
            marker=dict(symbol='line-ns-open', color='red', size=12),
            name='Fallido', legendgroup='Fallido',
            customdata=fails[['service_id','labor_id','map_start_point']].values,
            hovertemplate=(
                "Fallido<br>Servicio: %{customdata[0]}<br>"
                "Labor: %{customdata[1]}<br>"
                "Ubicación: %{customdata[2]}<extra></extra>"
            )
        ))

    # 6) Llegadas tarde
    late_tasks = (
        df_cleaned[df_cleaned['labor_category']=='VEHICLE_TRANSPORTATION']
          .sort_values('schedule_date')
          .groupby('service_id').first()
          .reset_index()
    )
    if not late_tasks.empty:
        late_tasks['late_dead'] = late_tasks['schedule_date'] + timedelta(minutes=tiempo_gracia)
        late = late_tasks[
            late_tasks[alfred_col].notna() &
            (late_tasks[start_col] > late_tasks['late_dead'])
        ].copy()
        if not late.empty:
            late['late_minutes'] = (
                (late[start_col] - late['late_dead']).dt.total_seconds() / 60
            ).round(1)
            fig.add_trace(go.Scatter(
                x=late[start_col],
                y=late[alfred_col].astype(str),
                mode='markers',
                marker=dict(symbol='x', size=12, color='black'),
                name='Llegada Tarde', legendgroup='Llegada Tarde',
                customdata=late[['service_id','labor_id','late_minutes']].values,
                hovertemplate=(
                    "Llegada Tarde<br>Servicio: %{customdata[0]}<br>"
                    "Labor: %{customdata[1]}<br>"
                    "Retraso: %{customdata[2]} min<extra></extra>"
                )
            ))
    fig.show()

    # 7) Gantt por servicios con detalle
    df_cleaned_plot = df_cleaned.dropna(subset=[start_col, end_col])
    if not df_cleaned_plot.empty:
        df_cleaned_plot['service_id_str'] = df_cleaned_plot['service_id'].astype(str)
        df_cleaned_plot['driver_label']   = df_cleaned_plot[alfred_col].fillna('on site').astype(str)
        fig2 = px.timeline(
            df_cleaned_plot.sort_values(['service_id_str', start_col]),
            x_start=start_col, x_end=end_col,
            y="service_id_str", color="labor_category",
            title=f"Detalle de Labors por Servicio ({day_str})",
            hover_data=["driver_label","labor_name"]
        )
        fig2.update_yaxes(autorange="reversed", type='category')
        fig2.update_layout(
            height=max(600, len(df_cleaned_plot['service_id_str'].unique()))
        )
        fig2.show()


def plot_horizontal_bar_from_tuples(
    data, 
    cities_df, 
    title="Servicios por Ciudad", 
    xlabel="Cantidad"
):
    """
    Genera un gráfico de barras horizontal a partir de una lista de tuplas (cod_ciudad, valor),
    mostrando el nombre real de la ciudad en vez del código.

    Parámetros
    ----------
    data : list of tuples
        Lista de tuplas donde el primer elemento es el código de ciudad (str o int)
        y el segundo es el valor numérico.
    cities_df : pd.DataFrame
        DataFrame con columnas ['cod_ciudad', 'ciudad'] para mapear códigos a nombres.
    title : str
        Título del gráfico.
    xlabel : str
        Etiqueta del eje X.
    """
    # Crear diccionario de mapeo
    city_map = dict(zip(cities_df["cod_ciudad"].astype(str), cities_df["ciudad"]))

    # Reemplazar códigos por nombres en los labels
    codes, values = zip(*data)
    labels = [city_map.get(str(code), str(code)) for code in codes]

    # Crear figura
    plt.figure(figsize=(9, 6))
    y = np.arange(len(labels))
    bars = plt.barh(y, values, color=plt.cm.plasma(np.linspace(0.3, 0.8, len(labels))), height=0.6)

    # Añadir valores al final de cada barra
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#333333"
        )

    # Estilo del gráfico
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.xlabel(xlabel, fontsize=13)
    plt.yticks(y, labels, fontsize=11)
    plt.xticks(fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    # Quitar bordes extra
    for spine in ["top", "right"]:
        plt.gca().spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_labor_duration_hist(
    df, 
    city, 
    labor_type, 
    shop=None,
    city_col="city", 
    labor_col="labor_type", 
    shop_col="shop", 
    start_col="labor_start_date", 
    end_col="labor_end_date", 
    bins=30
) -> None:
    """
    Plot histogram of labor durations for a given city and labor type,
    optionally filtered by shop.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with labor records.
    city : str
        City code or name to filter.
    labor_type : str
        Labor type to filter.
    shop : str, optional
        Shop to filter (default None = all shops).
    city_col, labor_col, shop_col : str
        Column names for city, labor type, and shop.
    start_col, end_col : str
        Column names for start and end datetime.
    bins : int
        Number of bins in the histogram.
    """

    # Ensure datetimes
    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    df[end_col] = pd.to_datetime(df[end_col], errors="coerce")

    # Compute durations
    df["duration_min"] = (df[end_col] - df[start_col]).dt.total_seconds() / 60
    df = df.dropna(subset=["duration_min"])

    # Remove durations > 1 day
    df = df[df["duration_min"] <= 1440]

    # Apply filters
    df_filtered = df[(df[city_col] == city) & (df[labor_col] == labor_type)]
    if shop is not None:
        df_filtered = df_filtered[df_filtered[shop_col] == shop]

    if df_filtered.empty:
        print("⚠️ No data available for the given filters.")
        return

    # Plot
    plt.figure(figsize=(8, 5))
    sns.histplot(df_filtered["duration_min"], bins=bins, kde=True, color="skyblue")

    title = f"Duration Distribution in {city} - {labor_type}"
    if shop:
        title += f" (Shop: {shop})"

    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel("Duration (minutes)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def plot_metrics_comparison(
    labors_hist_df: pd.DataFrame,
    moves_hist_df: pd.DataFrame,
    labors_algo_df: pd.DataFrame,
    moves_algo_df: pd.DataFrame,
    city: str,
    metricas,
    dist_dict: dict,
    fechas: tuple[str, str], 
    group_by=None,
    xaxis_mode: str = "date"
):
    """
    Genera gráficos comparativos de métricas entre solución real y algoritmo.
    Muestra de a dos métricas por figura (side-by-side), con leyenda debajo.

    Parámetros:
    - xaxis_mode: "date" para mostrar fechas reales,
                  "day" para mostrar etiquetas tipo 'day1', 'day2', ...
    """
    metrics_real_df = compute_metrics_with_moves(
        labors_hist_df, moves_hist_df, fechas, dist_dict,
        workday_hours=8, city=city, assignment_type='historic',
        skip_weekends=False, dist_method='haversine'
    )
    metrics_algo_df = compute_metrics_with_moves(
        labors_algo_df, moves_algo_df, fechas, dist_dict,
        workday_hours=8, city=city, assignment_type='algorithm',
        skip_weekends=False, dist_method='haversine'
    )

    if group_by is not None:
        metrics_real_df = add_aggregated_totals(metrics_real_df, group_by=group_by)
        metrics_algo_df = add_aggregated_totals(metrics_algo_df, group_by=group_by)

    # --- Labels ---
    labels = {
        "vt_count": "Labores tipo Vehicle Transportation",
        "num_drivers": "Número de conductores",
        "labores_por_conductor": "Labores por conductor",
        "utilizacion_promedio_%": "Utilización promedio (%)",
        "total_distance": "Distancia total (km)",
        "driver_move_distance": "Distancia recorrida por conductores (km)",
        "labor_extra_time": "Tiempo extra total por labor (min)",
        "driver_extra_time": "Tiempo extra total por conductor (min)",
    }

    if isinstance(metricas, str):
        metricas = [metricas]

    # --- Eje X ---
    x_real = pd.to_datetime(metrics_real_df["day"])
    x_alg = pd.to_datetime(metrics_algo_df["day"])

    if xaxis_mode == "day":
        # Crear etiquetas 'day1', 'day2', ...
        day_labels = [f"day{i+1}" for i in range(len(x_real))]
        x_real_display = day_labels
        x_alg_display = day_labels
    else:
        # Usar fechas reales
        x_real_display = x_real
        x_alg_display = x_alg

    # 🎨 Colors & styles
    color_real, color_alg = "#800080", "#17BECF"

    figs = []
    for i in range(0, len(metricas), 2):
        pair = metricas[i:i+2]
        fig = make_subplots(
            rows=1, cols=len(pair),
            subplot_titles=[labels.get(m, m) for m in pair]
        )

        for j, metrica in enumerate(pair, start=1):
            label = labels.get(metrica, metrica)

            # Serie real
            fig.add_trace(go.Scatter(
                x=x_real_display, y=metrics_real_df[metrica],
                mode="lines+markers",
                name=f"{label} (Real)",
                line=dict(color=color_real, dash="solid"),
                marker=dict(symbol="circle", size=8, color=color_real),
                legendgroup=f"{metrica}_real",
                showlegend=True
            ), row=1, col=j)

            # Serie algoritmo
            fig.add_trace(go.Scatter(
                x=x_alg_display, y=metrics_algo_df[metrica],
                mode="lines+markers",
                name=f"{label} (Algoritmo)",
                line=dict(color=color_alg, dash="dash"),
                marker=dict(symbol="x", size=8, color=color_alg),
                legendgroup=f"{metrica}_alg",
                showlegend=True
            ), row=1, col=j)

            fig.update_xaxes(title="Día", row=1, col=j)
            fig.update_yaxes(title=label, row=1, col=j)

        fig.update_layout(
            height=500,
            width=1000,
            title=f"Métricas en {get_city_name_from_code(city)} ({fechas[0]} a {fechas[1]})",
            margin=dict(t=100, b=100),
            legend=dict(
                orientation="h",
                yanchor="top", y=-0.25,
                xanchor="center", x=0.5
            )
        )

        fig.show()
        figs.append(fig)

    return figs


def plot_service_driver_distance(labors_hist_df: pd.DataFrame,
                                 moves_hist_df: pd.DataFrame,
                                 labors_algo_df: pd.DataFrame,
                                 moves_algo_df: pd.DataFrame,
                                 city: str,
                                 date: str,
                                 top_n: int = 15):

    def _compute_service_driver_distances(moves_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate driver move distances per service."""
        df = moves_df[moves_df["labor_category"] == "DRIVER_MOVE"].copy()
        if df.empty:
            return pd.DataFrame(columns=["service_id", "driver_move_distance"])
        return (
            df.groupby("service_id", as_index=False)["distance_km"]
            .sum()
            .rename(columns={"distance_km": "driver_move_distance"})
        )
    
    # --- Filter by city and date ---
    moves_hist_df = moves_hist_df[(moves_hist_df["city"] == city) &
                                  (moves_hist_df["schedule_date"].dt.strftime("%Y-%m-%d") == date)].copy()
    moves_algo_df = moves_algo_df[(moves_algo_df["city"] == city) &
                                  (moves_algo_df["schedule_date"].dt.strftime("%Y-%m-%d") == date)].copy()

    moves_hist_df['service_id'] = moves_hist_df['service_id'].astype(str)
    moves_algo_df['service_id'] = moves_algo_df['service_id'].astype(str)

    # --- Compute metrics ---
    hist_metrics = _compute_service_driver_distances(moves_hist_df)
    algo_metrics = _compute_service_driver_distances(moves_algo_df)

    # --- Select top services by algorithm ---
    top_services = algo_metrics.sort_values("driver_move_distance", ascending=False).head(top_n)["service_id"]
    hist_metrics = hist_metrics[hist_metrics["service_id"].isin(top_services)]
    algo_metrics = algo_metrics[algo_metrics["service_id"].isin(top_services)]

    merged = pd.merge(hist_metrics, algo_metrics, on="service_id", how="outer", suffixes=("_hist", "_algo")).fillna(0)
    merged = merged.set_index("service_id").loc[top_services]  # preserve order

    # --- Plot ---
    fig = go.Figure()

    colors = {"hist": "#800080", "algo": "#17BECF"}  # purple + teal

    fig.add_trace(go.Bar(
        x=merged.index, y=merged["driver_move_distance_hist"],
        name="Historic", marker_color=colors["hist"]
    ))
    fig.add_trace(go.Bar(
        x=merged.index, y=merged["driver_move_distance_algo"],
        name="Algorithm", marker_color=colors["algo"]
    ))

    fig.update_layout(
        barmode="group",
        title=f"Driver Move Distance per Service — City {codificacion_ciudades[city]}, Date {date}",
        xaxis_title="Service ID",
        yaxis_title="Driver Move Distance (km)",
        height=500, width=800,
        legend=dict(orientation="h", y=-0.2, x=0.3),
        margin=dict(t=50, b=50)
    )
    fig.update_xaxes(tickangle=-45)
    fig.show()


def prepare_labor_data(labors_static_df, labors_dynamic_df):
    """
    Combine static and dynamic labor dataframes, normalize columns, 
    and return the merged dataframe.
    """
    # Add a label column to distinguish static vs dynamic
    labors_static_df = labors_static_df.copy()
    labors_dynamic_df = labors_dynamic_df.copy()
    
    labors_static_df["labor_type"] = "Static"
    labors_dynamic_df["labor_type"] = "Dynamic"

    # Combine both
    all_labors_df = pd.concat([labors_static_df, labors_dynamic_df], ignore_index=True)
    all_labors_df["schedule_date"] = pd.to_datetime(all_labors_df["schedule_date"])

    return all_labors_df


def plot_labors(all_labors_df, selected_date_str):
    """
    Filter and plot labors by city and type for a given date or all dates.
    """
    # Handle 'ALL' selection
    if selected_date_str == "ALL":
        filtered = all_labors_df[
            all_labors_df["labor_category"] == "VEHICLE_TRANSPORTATION"
        ]
        title_date = "All Dates"
    else:
        selected_date = pd.to_datetime(selected_date_str).date()
        filtered = all_labors_df[
            (all_labors_df["schedule_date"].dt.date == selected_date)
            & (all_labors_df["labor_category"] == "VEHICLE_TRANSPORTATION")
        ]
        title_date = selected_date_str

    # Count per city and labor_type
    counts = (
        filtered.groupby(["city", "labor_type"])
        .size()
        .reset_index(name="count")
    )

    # Aesthetic color palette
    colors = ["#4F6D7A", "#E27D60"]  # muted teal and coral

    # Plot
    fig = px.bar(
        counts,
        x="city",
        y="count",
        color="labor_type",
        barmode="group",
        color_discrete_sequence=colors,
        title=f"Static vs Dynamic Labors ({title_date})",
        labels={"count": "Number of Labors", "city": "City", "labor_type": "Type"},
        template="plotly_white",
    )

    fig.update_layout(
        title_font_size=18,
        title_x=0.5,
        bargap=0.3,
        legend_title_text="Labor Type",
        font=dict(size=13),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )

    fig.show()


def make_histograms(df, created_col='created_at', schedule_col='schedule_date', unit='hours'):
    """
    Internal helper to build two histograms from the given dataframe.
    """
    # --- Safety checks ---
    if created_col not in df.columns or schedule_col not in df.columns:
        raise ValueError(f"Columns '{created_col}' and/or '{schedule_col}' not found in dataframe.")
    if unit not in ['hours', 'minutes']:
        raise ValueError("unit must be either 'hours' or 'minutes'")

    # --- 1. Distribution of order creation times (time of day) ---
    created_local = df[created_col].dt.tz_convert(df[created_col].dt.tz)
    created_hours = created_local.dt.hour + created_local.dt.minute / 60.0

    fig1 = px.histogram(
        x=created_hours,
        nbins=48,
        labels={'x': 'Hour of Day (local time)', 'y': 'Number of Orders'},
        title='Distribution of Order Creation Times (Local Time)',
        color_discrete_sequence=['#5DADE2'],  # soft blue
    )
    fig1.update_layout(
        bargap=0.05,
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(0, 25, 2)),
            title='Hour of Day (0–24)',
        ),
        yaxis_title='Count',
        template='plotly_white',
        title_x=0.5
    )

    # --- 2. Time difference between creation and schedule ---
    delta = df[schedule_col] - df[created_col]
    delta_hours = delta.dt.total_seconds() / 3600
    delta_vals = delta_hours if unit == 'hours' else delta_hours * 60
    
    fig2 = px.histogram(
        x=delta_vals,
        nbins=50,
        labels={'x': f'Time Between Creation and Schedule ({unit})', 'y': 'Number of Orders'},
        title=f'Distribution of Time Between Creation and Schedule ({unit})',
        color_discrete_sequence=['#F1948A'],  # soft coral
    )
    fig2.update_layout(
        bargap=0.05,
        template='plotly_white',
        title_x=0.5
    )

    return fig1, fig2