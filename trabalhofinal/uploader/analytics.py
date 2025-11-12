import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import io
import base64

MAX_CATEGORIES_FOR_PIE = 10
MIN_CATEGORIES_FOR_PIE = 2
UNIQUE_THRESHOLD_FOR_CATEGORICAL = 20


class DataAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df_raw = df
        self.df = self.clean_data(df.copy())
        self.numeric_cols = []
        self.categorical_cols = []
        self.date_cols = []
        self.geo_cols = []
        self._identify_column_types()

    def _fig_to_base64(self, fig) -> str:
        if isinstance(fig, plt.Figure):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            plt.close(fig)
            return f"data:image/png;base64,{img_str}"
        elif isinstance(fig, go.Figure):
            return fig.to_html(full_html=False, include_plotlyjs="cdn")
        return ""

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpeza básica: limpa nomes de colunas e remove duplicados.
        A lógica de conversão de tipo foi movida para _identify_column_types.
        O dropna() foi removido para permitir que os gráficos e o ML lidem com NaNs.
        """
        df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

        # Tenta converter colunas que parecem numéricas mas são 'object'
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="ignore")

        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    def _identify_column_types(self):
        """
        Identifica tipos de colunas e faz as conversões de tipo necessárias
        (ex: converter colunas de data que são 'object' para 'datetime').
        """
        geo_keywords = [
            "latitude",
            "longitude",
            "lat",
            "lon",
            "cep",
            "cidade",
            "estado",
            "pais",
        ]
        date_keywords = ["data", "date", "ano", "year", "time", "timestamp"]

        for col in self.df.columns:
            dtype = self.df[col].dtype
            nunique = self.df[col].nunique()

            # Tenta converter para data se for 'object' e tiver keywords
            if dtype == "object" and any(keyword in col for keyword in date_keywords):
                temp_series = pd.to_datetime(self.df[col], errors="coerce")
                if not pd.isna(temp_series).all():
                    self.df[col] = temp_series
                    dtype = self.df[col].dtype

            if any(keyword in col for keyword in geo_keywords):
                if col not in self.geo_cols:
                    self.geo_cols.append(col)

            if pd.api.types.is_datetime64_any_dtype(dtype):
                if col not in self.date_cols:
                    self.date_cols.append(col)

            elif pd.api.types.is_numeric_dtype(dtype) and (
                nunique >= UNIQUE_THRESHOLD_FOR_CATEGORICAL or nunique == 0
            ):
                if col not in self.numeric_cols:
                    self.numeric_cols.append(col)

            elif pd.api.types.is_string_dtype(dtype) or (
                pd.api.types.is_numeric_dtype(dtype)
                and nunique < UNIQUE_THRESHOLD_FOR_CATEGORICAL
            ):
                if col not in self.categorical_cols:
                    self.categorical_cols.append(col)

        # Garante que as listas não tenham sobreposição
        self.numeric_cols = [
            c
            for c in self.numeric_cols
            if c not in self.categorical_cols
            and c not in self.date_cols
            and c not in self.geo_cols
        ]
        self.categorical_cols = [
            c for c in self.categorical_cols if c not in self.date_cols
        ]
        self.geo_cols = [c for c in self.geo_cols if c not in self.date_cols]

    def generate_basic_plots(self) -> list[dict]:
        plots = []

        for col in self.categorical_cols:
            try:
                if self.df[col].nunique() == 0:
                    continue

                # Gráfico de Barras para Top 20
                counts = self.df[col].value_counts().nlargest(20).sort_values()
                if not counts.empty:
                    fig_bar = px.bar(
                        counts, orientation="h", title=f'Contagem de "{col}" (Top 20)'
                    )
                    plots.append(
                        {
                            "section": "Análise Categórica",
                            "title": f'Contagem por "{col}"',
                            "html": self._fig_to_base64(fig_bar),
                        }
                    )

                # Gráfico de Pizza se houver poucas categorias
                nunique = self.df[col].nunique()
                if MIN_CATEGORIES_FOR_PIE <= nunique <= MAX_CATEGORIES_FOR_PIE:
                    fig_pie = px.pie(
                        self.df, names=col, title=f'Distribuição em Pizza de "{col}"'
                    )
                    plots.append(
                        {
                            "section": "Análise Categórica",
                            "title": f'Pizza de "{col}"',
                            "html": self._fig_to_base64(fig_pie),
                        }
                    )
            except Exception as e:
                print(f"Error generating basic plot for {col}: {e}")

        for col in self.numeric_cols:
            try:
                # Histograma e Boxplot
                if not self.df[col].empty:
                    fig_hist = px.histogram(
                        self.df,
                        x=col,
                        marginal="box",
                        title=f'Histograma e Boxplot de "{col}"',
                    )
                    plots.append(
                        {
                            "section": "Análise Numérica",
                            "title": f'Distribuição de "{col}"',
                            "html": self._fig_to_base64(fig_hist),
                        }
                    )

                # Tabela de Estatísticas
                stats_html = (
                    self.df[col]
                    .describe()
                    .to_frame()
                    .to_html(classes="table table-striped table-hover")
                )
                plots.append(
                    {
                        "section": "Análise Numérica",
                        "title": f'Estatísticas Descritivas para "{col}"',
                        "html": stats_html,
                    }
                )
            except Exception as e:
                print(f"Error generating basic plot for {col}: {e}")

        return plots

    def generate_advanced_plots(self) -> list[dict]:
        plots = []

        for col in self.numeric_cols:
            try:
                if not self.df[col].empty:
                    fig_violin = px.violin(
                        self.df,
                        y=col,
                        box=True,
                        points="all",
                        title=f'Distribuição (Violin Plot) de "{col}"',
                    )
                    plots.append(
                        {
                            "section": "Análise Avançada Univariada",
                            "title": f'Violin Plot de "{col}"',
                            "html": self._fig_to_base64(fig_violin),
                        }
                    )
            except Exception as e:
                print(f"Error generating violin plot for {col}: {e}")

        if len(self.numeric_cols) > 1:
            try:
                corr = self.df[self.numeric_cols].corr()
                fig_heatmap = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    title="Heatmap de Correlação Numérica",
                )
                plots.append(
                    {
                        "section": "Análise Avançada Bivariada",
                        "title": "Heatmap de Correlação",
                        "html": self._fig_to_base64(fig_heatmap),
                    }
                )

                if len(self.numeric_cols) > 1:
                    sorted_corr = corr.abs().unstack().sort_values(ascending=False)
                    sorted_corr = sorted_corr[sorted_corr < 1]
                    top_pairs = sorted_corr.head(3).index.tolist()

                    for pair in top_pairs:
                        col1, col2 = pair
                        if col1 in self.df.columns and col2 in self.df.columns:
                            fig_scatter = px.scatter(
                                self.df,
                                x=col1,
                                y=col2,
                                trendline="ols",
                                title=f"Correlação: {col1} vs {col2}",
                            )
                            plots.append(
                                {
                                    "section": "Análise Avançada Bivariada",
                                    "title": f"Scatter: {col1} vs {col2}",
                                    "html": self._fig_to_base64(fig_scatter),
                                }
                            )
            except Exception as e:
                print(f"Error generating advanced bivariate plots: {e}")

        return plots

    def generate_geo_visualization(self) -> dict | None:
        lat_col = next((c for c in self.df.columns if "lat" in c), None)
        lon_col = next((c for c in self.df.columns if "lon" in c or "lng" in c), None)

        if (
            lat_col
            and lon_col
            and pd.api.types.is_numeric_dtype(self.df[lat_col])
            and pd.api.types.is_numeric_dtype(self.df[lon_col])
        ):
            try:
                color_col = self.categorical_cols[0] if self.categorical_cols else None

                fig_map = px.scatter_mapbox(
                    self.df,
                    lat=lat_col,
                    lon=lon_col,
                    color=color_col,
                    mapbox_style="carto-positron",
                    zoom=2,
                    title="Visualização Geográfica de Dados",
                )
                if color_col:
                    fig_map.update_layout(legend_title=color_col)
                return {
                    "section": "Análise Geográfica",
                    "title": "Mapa de Dispersão Geográfica",
                    "html": self._fig_to_base64(fig_map),
                }
            except Exception as e:
                print(f"Error generating geo map: {e}")

        geo_col_bar = next(
            (c for c in self.geo_cols if c in self.categorical_cols), None
        )
        if geo_col_bar:
            try:
                top_items = (
                    self.df[geo_col_bar].value_counts().nlargest(15).sort_values()
                )
                if not top_items.empty:
                    fig_bar_geo = px.bar(
                        top_items,
                        x=top_items.values,
                        y=top_items.index,
                        orientation="h",
                        title=f'Top 15 Localidades em "{geo_col_bar}"',
                    )
                    return {
                        "section": "Análise Geográfica",
                        "title": f'Contagem por "{geo_col_bar}"',
                        "html": self._fig_to_base64(fig_bar_geo),
                    }
            except Exception as e:
                print(f"Error generating geo bar chart: {e}")

        return None

    def generate_temporal_plots(self) -> list[dict]:
        plots = []
        for col in self.date_cols:
            try:
                df_temp = self.df.copy()
                df_temp[col] = pd.to_datetime(df_temp[col], errors="coerce").dropna()

                if df_temp.empty:
                    continue

                time_series = df_temp.set_index(col).resample("D").size()
                time_series = time_series[time_series > 0]

                if time_series.empty:
                    continue

                fig_line = px.line(
                    time_series,
                    x=time_series.index,
                    y=time_series.values,
                    title=f"Evolução Temporal Diária ({col})",
                    markers=True,
                )
                fig_line.update_layout(xaxis_title="Data", yaxis_title="Contagem")
                plots.append(
                    {
                        "section": "Análise Temporal",
                        "title": f"Evolução por Data ({col})",
                        "html": self._fig_to_base64(fig_line),
                    }
                )

                if len(time_series) > 7:
                    time_series_ma = time_series.rolling(window=7).mean()
                    fig_ma = go.Figure()
                    fig_ma.add_trace(
                        go.Scatter(
                            x=time_series.index,
                            y=time_series.values,
                            mode="lines",
                            name="Contagem Diária",
                        )
                    )
                    fig_ma.add_trace(
                        go.Scatter(
                            x=time_series_ma.index,
                            y=time_series_ma.values,
                            mode="lines",
                            name="Média Móvel (7 dias)",
                        )
                    )
                    fig_ma.update_layout(
                        title=f"Tendência com Média Móvel ({col})",
                        xaxis_title="Data",
                        yaxis_title="Contagem",
                    )
                    plots.append(
                        {
                            "section": "Análise Temporal",
                            "title": f"Tendência com Média Móvel ({col})",
                            "html": self._fig_to_base64(fig_ma),
                        }
                    )

            except Exception as e:
                print(f"Error generating temporal plot for {col}: {e}")

        return plots
