from django.shortcuts import render, redirect
import pandas as pd
import io
import os
from django.conf import settings
from django.core.files.storage import default_storage
from .analytics import DataAnalyzer
from .ml_models import run_ml_task


def upload_file(request):
    if request.method == "POST":
        f = request.FILES.get("csv_file") or request.FILES.get("file")
        if not f:
            return render(
                request, "uploader/upload.html", {"error": "Envie um arquivo .csv."}
            )

        try:
            save_path = os.path.join('uploads', f.name)
            actual_path = default_storage.save(save_path, f)
            full_fs_path = os.path.join(settings.MEDIA_ROOT, actual_path)

            df = pd.read_csv(full_fs_path, encoding="utf-8", on_bad_lines="skip")
            
            request.session["file_path"] = actual_path
            request.session["df_columns"] = list(df.columns)
            request.session.pop("dataframe", None)
            
            try:
                upload_dir_name = 'uploads'
                max_files = 3
                _, filenames = default_storage.listdir(upload_dir_name)
                csv_files = [os.path.join(upload_dir_name, f) for f in filenames if f.lower().endswith('.csv')]
                
                if len(csv_files) > max_files:
                    files_with_mtime = []
                    for path in csv_files:
                        try:
                            full_file_path = os.path.join(settings.MEDIA_ROOT, path)
                            files_with_mtime.append((path, os.path.getmtime(full_file_path)))
                        except (IOError, FileNotFoundError):
                            continue
                    
                    files_with_mtime.sort(key=lambda x: x[1])
                    
                    num_to_delete = len(files_with_mtime) - max_files
                    files_to_delete = files_with_mtime[:num_to_delete]
                    
                    for path, _ in files_to_delete:
                        try:
                            default_storage.delete(path)
                        except (IOError, FileNotFoundError):
                            continue
            except (IOError, FileNotFoundError) as e:
                print(f"Erro de E/S na limpeza de arquivos: {e}")
                pass
            except Exception as e:
                print(f"Erro inesperado durante a limpeza de arquivos: {e}")
                pass

            return redirect("analysis")
        except Exception as e:
            if 'actual_path' in locals() and default_storage.exists(actual_path):
                default_storage.delete(actual_path)
            
            return render(
                request,
                "uploader/upload.html",
                {"error": f"Erro ao processar o arquivo: {e}"},
            )

    return render(request, "uploader/upload.html")


def analysis_view(request):
    file_path = request.session.get("file_path")
    if not file_path:
        return render(
            request,
            "uploader/analysis.html",
            {
                "plots": [],
                "error": "Nenhum dado para analisar. Por favor, faça o upload de um arquivo CSV.",
            },
        )

    try:
        full_fs_path = os.path.join(settings.MEDIA_ROOT, file_path)
        
        if not default_storage.exists(file_path):
             return render(
                request,
                "uploader/analysis.html",
                {
                    "plots": [],
                    "error": "Arquivo não encontrado ou expirado. Por favor, faça o upload novamente.",
                },
            )

        df = pd.read_csv(full_fs_path, encoding="utf-8", on_bad_lines="skip")

        if df.empty:
            return render(
                request,
                "uploader/analysis.html",
                {"plots": [], "error": "O DataFrame está vazio após o carregamento."},
            )

        analyzer = DataAnalyzer(df)

        if analyzer.df.empty:
            return render(
                request,
                "uploader/analysis.html",
                {
                    "plots": [],
                    "error": "O DataFrame ficou vazio após a limpeza de dados (ex: remoção de valores nulos).",
                },
            )

        plots = []
        plots.extend(analyzer.generate_basic_plots())
        plots.extend(analyzer.generate_advanced_plots())
        geo_plot = analyzer.generate_geo_visualization()
        if geo_plot:
            plots.append(geo_plot)
        temporal_plots = analyzer.generate_temporal_plots()
        plots.extend(temporal_plots)

        grouped_plots = {}
        for plot in plots:
            section = plot.get("section", "Geral")
            if section not in grouped_plots:
                grouped_plots[section] = []
            grouped_plots[section].append(plot)

        return render(
            request, "uploader/analysis.html", {"grouped_plots": grouped_plots}
        )

    except Exception as e:
        return render(
            request,
            "uploader/analysis.html",
            {"plots": [], "error": f"Ocorreu um erro durante a análise: {e}"},
        )


def prediction_view(request):
    cols = request.session.get("df_columns") or []

    input_fields = []
    if cols:
        feature_cols = cols[:-1]
        for c in feature_cols:
            input_fields.append({"name": c, "label": c.capitalize().replace("_", " ")})
    else:
        input_fields = [
            {"name": "idade", "label": "Idade", "type": "number", "placeholder": "30"},
            {
                "name": "genero",
                "label": "Gênero (texto)",
                "type": "text",
                "placeholder": "Ação",
            },
            {
                "name": "orcamento",
                "label": "Orçamento (R$)",
                "type": "number",
                "placeholder": "10000000",
            },
        ]

    ctx = {"input_fields": input_fields}

    if request.method == "POST":
        file_path = request.session.get("file_path")
        if not file_path:
            ctx["prediction"] = {
                "output": "Sessão expirada. Faça upload do CSV novamente.",
                "metrics": "",
            }
            return render(request, "uploader/prediction.html", ctx)

        try:
            full_fs_path = os.path.join(settings.MEDIA_ROOT, file_path)
            if not default_storage.exists(file_path):
                 raise Exception("Arquivo não encontrado ou expirado. Faça o upload novamente.")
            
            df = pd.read_csv(full_fs_path, encoding="utf-8", on_bad_lines="skip")
            analyzer = DataAnalyzer(df)
            df_clean = analyzer.df

            if df_clean.empty or len(df_clean.columns) < 2:
                ctx["prediction"] = {
                    "output": "Erro: Os dados limpos estão vazios ou não têm colunas suficientes (mínimo 2).",
                    "metrics": "",
                }
                return render(request, "uploader/prediction.html", ctx)

        except Exception as e:
            ctx["prediction"] = {
                "output": f"Erro ao ler dados da sessão: {e}",
                "metrics": "",
            }
            return render(request, "uploader/prediction.html", ctx)

        action = request.POST.get("action")
        modelo = request.POST.get("modelo")

        if not modelo:
            ctx["prediction"] = {"output": "Modelo não selecionado.", "metrics": ""}
            return render(request, "uploader/prediction.html", ctx)

        hps = {k[3:]: v for k, v in request.POST.items() if k.startswith("hp_")}
        xs = {k: v for k, v in request.POST.items() if k.startswith("X_")}

        try:
            prediction_result = run_ml_task(df_clean, modelo, hps, xs, action)
            ctx["prediction"] = prediction_result
        except Exception as e:
            ctx["prediction"] = {
                "output": f"Erro inesperado na execução do ML: {e}",
                "metrics": "N/A",
            }

    return render(request, "uploader/prediction.html", ctx)