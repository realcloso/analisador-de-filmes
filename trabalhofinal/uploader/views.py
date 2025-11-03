from django.shortcuts import render, redirect
import pandas as pd
import base64

def upload_file(request):
    if request.method == 'POST':
        f = request.FILES.get('csv_file') or request.FILES.get('file')
        if not f:
            return render(request, 'uploader/upload.html', {'error': 'Envie um arquivo .csv.'})

        try:
            df = pd.read_csv(f)
            request.session['dataframe'] = df.to_json(orient='split')
            request.session['df_columns'] = list(df.columns)
            request.session['has_df'] = True  # flag para o mock
            return redirect('analysis')
        except Exception as e:
            return render(request, 'uploader/upload.html', {'error': str(e)})

    return render(request, 'uploader/upload.html')


# ====== ANALYSIS (com gráficos MOCK) ======
def analysis_view(request):
    plots = []
    if request.session.get('has_df'):
        svg_bar = """
        <svg xmlns='http://www.w3.org/2000/svg' width='480' height='220'>
          <rect width='480' height='220' fill='#1a1e35'/>
          <rect x='40' y='120' width='60' height='80' fill='#6c7bff'/>
          <rect x='140' y='80' width='60' height='120' fill='#33d1ff'/>
          <rect x='240' y='60' width='60' height='140' fill='#6c7bff'/>
          <rect x='340' y='150' width='60' height='50' fill='#33d1ff'/>
          <text x='40' y='30' fill='#eef0ff' font-size='16'>Exemplo (mock) - Barras</text>
        </svg>
        """.strip()

        svg_heat = """
        <svg xmlns='http://www.w3.org/2000/svg' width='480' height='220'>
          <defs>
            <linearGradient id='g' x1='0' x2='1'>
              <stop offset='0%' stop-color='#6c7bff'/>
              <stop offset='100%' stop-color='#33d1ff'/>
            </linearGradient>
          </defs>
          <rect width='480' height='220' fill='#1a1e35'/>
          <rect x='40'  y='40'  width='120' height='60' fill='url(#g)' opacity='0.9'/>
          <rect x='180' y='40'  width='120' height='60' fill='url(#g)' opacity='0.7'/>
          <rect x='320' y='40'  width='120' height='60' fill='url(#g)' opacity='0.5'/>
          <rect x='40'  y='120' width='120' height='60' fill='url(#g)' opacity='0.6'/>
          <rect x='180' y='120' width='120' height='60' fill='url(#g)' opacity='0.85'/>
          <rect x='320' y='120' width='120' height='60' fill='url(#g)' opacity='0.4'/>
          <text x='40' y='30' fill='#eef0ff' font-size='16'>Exemplo (mock) - Heatmap</text>
        </svg>
        """.strip()

        def svg_to_dataurl(svg: str) -> str:
            return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

        plots = [
            {"section": "basic", "title": "Gêneros (mock)", "img": svg_to_dataurl(svg_bar)},
            {"section": "advanced", "title": "Correlação (mock)", "img": svg_to_dataurl(svg_heat)},
        ]

    return render(request, 'uploader/analysis.html', {"plots": plots})


def prediction_view(request):
    cols = request.session.get('df_columns') or []
    if cols:
        input_fields = [{"name": c, "label": c.capitalize(), "type": "text", "placeholder": ""} for c in cols[:6]]
    else:
        input_fields = [
            {"name": "idade", "label": "Idade", "type": "number", "placeholder": "30"},
            {"name": "genero", "label": "Gênero (texto)", "type": "text", "placeholder": "Ação"},
            {"name": "orcamento", "label": "Orçamento (R$)", "type": "number", "placeholder": "10000000"},
        ]

    ctx = {"input_fields": input_fields}

    if request.method == 'POST':
        action = request.POST.get('action')         
        modelo = request.POST.get('modelo')          

        if not modelo:
            ctx["prediction"] = {"output": "Modelo não selecionado.", "metrics": ""}
            return render(request, 'uploader/prediction.html', ctx)

        hps = {k[3:]: v for k, v in request.POST.items() if k.startswith("hp_")}
        xs  = {k[2:]: v for k, v in request.POST.items() if k.startswith("X_")}

        if action == 'retrain':
            output = f"Modelo {modelo} re-treinado com {len(xs)} features e {len(hps)} hiperparâmetros. (mock)"
            metrics = "acc=0.90 | f1=0.88 (mock)"
        else:
            output = f"Predição (mock) com {modelo}: classe='Sucesso' / score=0.82"
            metrics = "acc=0.88 | f1=0.86 (mock)"

        ctx["prediction"] = {"output": output, "metrics": metrics}

    return render(request, 'uploader/prediction.html', ctx)
