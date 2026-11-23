@echo off
REM Builds dist\NeuralGuitarIRStudio\NeuralGuitarIRStudio.exe (needs Python only on the BUILD machine)
py -3.11 -m venv .venv && call .venv\Scripts\activate
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --windowed --name NeuralGuitarIRStudio --collect-all pyqtgraph --collect-all sounddevice launcher.py
echo Done: dist\NeuralGuitarIRStudio\NeuralGuitarIRStudio.exe
