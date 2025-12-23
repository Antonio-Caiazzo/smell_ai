import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Union, Optional
import sys
import time

import pandas as pd
import pytest
import requests
import os

# Base directory che contiene le cartelle TC_01 .. TC_20
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
START_SCRIPT = REPO_ROOT / "start_minimal_services.sh"
PIDS_FILE = REPO_ROOT / "minimal_services.pids"
GATEWAY_URL = "http://localhost:8000"


@pytest.fixture(scope="session", autouse=True)
def manage_services():
    """
    Automatically starts minimal services before system tests and stops them after.
    Defined here to avoid external conftest.py.
    """
    print("\n[setup] Starting minimal services from test_system_script.py...")
    
    if not START_SCRIPT.exists():
        print(f"[setup] Warning: {START_SCRIPT} not found. Skipping auto-start.")
        yield
        return

    # Ensure script is executable
    START_SCRIPT.chmod(START_SCRIPT.stat().st_mode | 0o111)

    # Start services
    subprocess.run([str(START_SCRIPT)], cwd=str(REPO_ROOT), check=True)

    # Wait for Gateway to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            requests.get(f"{GATEWAY_URL}/", timeout=1)
            print("\n[setup] Gateway is ready!")
            break
        except requests.exceptions.RequestException:
            if i == max_retries - 1:
                print("\n[setup] Gateway failed to start within timeout.")
            time.sleep(1)

    yield

    # Teardown
    print("\n[teardown] Stopping services...")
    if PIDS_FILE.exists():
        subprocess.run(["pkill", "-F", str(PIDS_FILE)], check=False)
        try:
            PIDS_FILE.unlink()
        except FileNotFoundError:
            pass
TestConfig = Dict[str, Union[str, bool, int, List[str]]]


TEST_CASES: Dict[str, TestConfig] = {
    # ================== ERROR CASES (TC_01..TC_05) ==================

    # TC_01
    # NF1, EF0, NP1, SP1, NCS0, TCS0, ME2, EP2, NW0, RES2, OUT1
    # Oracolo: il tool segnala che non ci sono file in input da analizzare
    #          e non esegue alcuna analisi.
    "TC_01": {
        "description": "Assenza di file di input: nessuna analisi, errore segnalato.",
        "expected_error": True,
        "expected_smells": None,
        "parallel": False,   # EP2
        "max_walkers": 5,    # ignorato perché parallel=False
        "resume": False,     # RES2
        "multiple": False,   # NP1
    },

    # TC_02
    # NF2, EF1, NP1, SP3, NCS0, TCS0, ME2, EP2, NW0, RES2, OUT1
    # Oracolo: errore sulla struttura del progetto, sottocartelle inaccessibili.
    "TC_02": {
        "description": "Struttura del progetto con sottocartelle inaccessibili: errore segnalato.",
        "expected_error": True,
        "expected_smells": None,
        "parallel": False,   # EP2
        "max_walkers": 5,
        "resume": False,     # RES2
        "multiple": False,   # NP1
        # Path (relativo a tc_path) che rappresenta la sottocartella da rendere inaccessibile.
        # Adatta questo nome in base alla struttura reale della cartella TC_02.
        "unreadable_paths": ["MockDirectoryNotAccessible"],
    },

    # TC_03
    # NF2, EF1, NP1, SP1, NCS0, TCS0, ME2, EP1, NW1, RES2, OUT1
    # Oracolo: numero di walkers non valido (<=0), nessuna analisi.
    "TC_03": {
        "description": "Numero di walkers non valido (<=0) con parallelismo attivo: errore.",
        "expected_error": True,
        "expected_smells": None,
        "parallel": True,    # EP1
        "max_walkers": 0,    # NW1 (<=0) -> errore
        "resume": False,     # RES2
        "multiple": False,   # NP1
    },

    # TC_04
    # NF2, EF1, NP1, SP1, NCS0, TCS0, ME2, EP2, NW0, RES2, OUT2
    # Oracolo: percorso di output mancante o non accessibile.
    "TC_04": {
        "description": "Percorso di output non accessibile: errore.",
        "expected_error": True,
        "expected_smells": None,
        "parallel": False,   # EP2
        "max_walkers": 5,
        "resume": False,     # RES2
        "multiple": False,   # NP1
        # flag speciale per rendere inaccessibile la cartella di output generata
        "lock_output": True,
    },

    # TC_05
    # NF2, EF2, NP1, SP1, NCS0, TCS0, ME2, EP2, NW0, RES2, OUT1
    # Oracolo: nessun file .py, il tool segnala l’errore e non analizza.
    "TC_05": {
        "description": "Nessun file Python da analizzare: errore e nessuna analisi.",
        "expected_error": True,
        "expected_smells": None,
        "parallel": False,   # EP2
        "max_walkers": 5,
        "resume": False,     # RES2
        "multiple": False,   # NP1
    },

    # ================== SUCCESS CASES – NCS = 0 (TC_06..TC_08) ==================

    # TC_06
    # NF3, EF1, NP1, SP1, NCS1 (0 smells), TCS0, ME2, EP2, NW0, RES2, OUT1
    # Oracolo: singolo progetto, nessun code smell.
    "TC_06": {
        "description": "Analisi singolo progetto, nessun code smell.",
        "expected_error": False,
        "expected_smells": 0,  # NCS1 -> 0 smell
        "parallel": False,     # EP2
        "max_walkers": 5,
        "resume": False,       # RES2
        "multiple": False,     # NP1
    },

    # TC_07
    # NF3, EF1, NP2, SP1, NCS1 (0 smells), TCS0, ME2, EP2, NW0, RES2, OUT1
    # Oracolo: multi progetto, nessun code smell.
    "TC_07": {
        "description": "Analisi multi-progetto, nessun code smell.",
        "expected_error": False,
        "expected_smells": 0,
        "parallel": False,     # EP2
        "max_walkers": 5,
        "resume": False,       # RES2
        "multiple": True,      # NP2
    },

    # TC_08
    # NF3, EF1, NP1, SP2, NCS1 (0 smells), TCS0, ME2, EP2, NW0, RES2, OUT1
    # Oracolo: progetto singolo, struttura annidata, nessun code smell.
    "TC_08": {
        "description": "Analisi singolo progetto annidato, nessun code smell.",
        "expected_error": False,
        "expected_smells": 0,
        "parallel": False,     # EP2
        "max_walkers": 5,
        "resume": False,       # RES2
        "multiple": False,     # NP1
    },

    # ================== SUCCESS CASES – SMELL PRESENTI (TC_09..TC_16) ==================

    # TC_09
    # NF2, EF1, NP1, SP1, NCS2 (1 smell), TCS1 (generico),
    # ME2, EP2, NW0, RES1, OUT1
    "TC_09": {
        "description": "Analisi singolo progetto: 1 code smell generico.",
        "expected_error": False,
        "expected_smells": 1,
        "parallel": False,     # EP2
        "max_walkers": 5,
        "resume": True,        # RES1
        "multiple": False,     # NP1
    },

    # TC_10
    # NF2, EF1, NP1, SP1, NCS2 (1 smell), TCS2 (API-specific),
    # ME2, EP2, NW0, RES1, OUT1
    "TC_10": {
        "description": "Analisi singolo progetto: 1 code smell API-specific.",
        "expected_error": False,
        "expected_smells": 1,
        "parallel": False,     # EP2
        "max_walkers": 5,
        "resume": True,        # RES1
        "multiple": False,     # NP1
    },

    # TC_11
    # NF2, EF1, NP1, SP1, NCS3 (>1), TCS1 (generici),
    # ME2, EP2, NW0, RES1, OUT1
    "TC_11": {
        "description": "Analisi singolo progetto: >1 code smell generico.",
        "expected_error": False,
        "expected_smells": ">=2",  # NCS3 -> >1 smell
        "parallel": False,         # EP2
        "max_walkers": 5,
        "resume": True,            # RES1
        "multiple": False,         # NP1
    },

    # TC_12
    # NF2, EF1, NP1, SP1, NCS3 (>1), TCS2 (API-specific),
    # ME2, EP2, NW0, RES1, OUT1
    "TC_12": {
        "description": "Analisi singolo progetto: >1 code smell API-specific.",
        "expected_error": False,
        "expected_smells": ">=2",
        "parallel": False,         # EP2
        "max_walkers": 5,
        "resume": True,            # RES1
        "multiple": False,         # NP1
    },

    # TC_13
    # NF2, EF1, NP1, SP1, NCS3 (>1), TCS3 (misto),
    # ME2, EP2, NW0, RES1, OUT1
    "TC_13": {
        "description": "Analisi singolo progetto: >1 code smell misto.",
        "expected_error": False,
        "expected_smells": ">=2",
        "parallel": False,         # EP2
        "max_walkers": 5,
        "resume": True,            # RES1
        "multiple": False,         # NP1
    },

    # TC_14
    # NF2, EF1, NP1, SP1, NCS2 (1 smell), TCS1,
    # ME2, EP1, NW2 (<5), RES1, OUT1
    "TC_14": {
        "description": "Analisi parallela: 1 code smell generico, walkers < 5.",
        "expected_error": False,
        "expected_smells": 1,
        "parallel": True,          # EP1
        "max_walkers": 3,          # NW2 -> <5
        "resume": True,            # RES1
        "multiple": False,         # NP1
    },

    # TC_15
    # NF2, EF1, NP1, SP1, NCS2 (1 smell), TCS1,
    # ME2, EP1, NW3 (=5), RES1, OUT1
    "TC_15": {
        "description": "Analisi parallela: 1 code smell generico, walkers = 5.",
        "expected_error": False,
        "expected_smells": 1,
        "parallel": True,          # EP1
        "max_walkers": 5,          # NW3 -> =5
        "resume": True,            # RES1
        "multiple": False,         # NP1
    },

    # TC_16
    # NF2, EF1, NP1, SP1, NCS2 (1 smell), TCS1,
    # ME2, EP1, NW4 (>5), RES1, OUT1
    "TC_16": {
        "description": "Analisi parallela: 1 code smell generico, walkers > 5.",
        "expected_error": False,
        "expected_smells": 1,
        "parallel": True,          # EP1
        "max_walkers": 6,          # NW4 -> >5
        "resume": True,            # RES1
        "multiple": False,         # NP1
    },

    # ================== WEBAPP CASES (TC_17..TC_20) ==================

    # TC_17 
    # NF2, EF1, NP1, SP1, NCS1,
    # TCS0, ME1, EP2, NW0, RES0,
    # OUT1, DB1, EG1
    # Servizi raggiungibili e richiesta completata correttamente (HTTP 2xx).
    # Usa static analysis.
    "TC_17": {
        "description": "WebApp: servizi raggiungibili e analisi OK.",
        "type": "WEBAPP",
        "endpoint": "/api/detect_smell_static",
        "expected_status": "2xx",
        "expected_error": False,
    },


    # TC_18 
    # NF2, EF1, NP1, SP1, NCS0,
    # TCS0, ME1, EP2, NW0, RES0,
    # OUT1, DB2, EG0
    # Almeno un servizio backend non raggiungibile -> errore.
    # Il gateway ritorna 200 con {"success": False}.
    "TC_18": {
        "description": "WebApp: backend non raggiungibile (wrapped error).",
        "type": "WEBAPP",
        "endpoint": "/api/detect_smell_ai",
        "expected_status": "200_error_wrapped", 
        "expected_error": True,
    },

    # TC_19 
    # NF2, EF1, NP1, SP1, NCS0,
    # TCS0, ME1, EP2, NW0, RES0,
    # OUT1, DB1, EG2
    # Backend raggiungibile ma errore di validazione richiesta -> HTTP 4xx .
    # Il gateway wrappa in 200 OK.
    "TC_19": {
        "description": "WebApp: errore di validazione input (wrapped error).",
        "type": "WEBAPP",
        "endpoint": "/api/detect_smell_static",
        "expected_status": "200_validation_error",
        "invalid_request": True,
        "expected_error": True,
    },

    # TC_20 
    # NF2, EF1, NP1, SP1, NCS0,
    # TCS0, ME1, EP2, NW0, RES0,
    # OUT1, DB1, EG3
    # Backend raggiungibile ma errore infrastrutturale o timeout -> HTTP 5xx o timeout.
    "TC_20": {
        "description": "WebApp: errore infrastrutturale o timeout (HTTP 5xx).",
        "type": "WEBAPP",
        "endpoint": "/api/detect_smell_static",
        "expected_status": "5xx",
        "timeout_test": True,
        "expected_error": True,
    },
}


def make_unreadable(path: Path) -> Optional[int]:
    """
    Rende il path non leggibile/non scrivibile.
    Restituisce il mode originale, così da poterlo ripristinare.
    Se il path non esiste, restituisce None.
    """
    if not path.exists():
        return None
    current_mode = path.stat().st_mode
    path.chmod(0)
    return current_mode


def make_readable(path: Path, mode: Optional[int]) -> None:
    """
    Ripristina i permessi originali se mode è valorizzato.
    """
    if mode is None:
        return
    if path.exists():
        path.chmod(mode)


def list_test_cases() -> List[str]:
    """
    Elenca le directory TC_01..TC_16 in BASE_DIR, ordinate numericamente.
    """
    cases: List[str] = []
    if not BASE_DIR.exists():
        return cases

    for entry in BASE_DIR.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith("TC_"):
            continue
        try:
            number = int(name.split("_")[-1])
        except ValueError:
            continue
        if 1 <= number <= 20:
            cases.append(name)

    cases.sort(key=lambda x: int(x.split("_")[-1]))
    return cases


@pytest.mark.parametrize("tc_dir", list_test_cases())
def test_system_case(tc_dir: str) -> None:
    """
    Esegue il test di sistema per la cartella tc_dir (es. 'TC_02'),
    configurando parametri e oracolo secondo il documento di system testing.
    """
    tc_path = BASE_DIR / tc_dir
    config = TEST_CASES[tc_dir]

    output_dir = BASE_DIR / "output" / tc_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 0) Check if WEBAPP or CLI
    if config.get("type") == "WEBAPP":
        gateway_url = os.environ.get("CODESMILE_GATEWAY_URL", "http://localhost:8000")
        
        # Read the first file in tc_path to use as code_snippet
        code_snippet = ""
        if tc_path.exists():
            for f in tc_path.rglob("*"):
                if f.is_file() and f.suffix == ".py":
                    try:
                        code_snippet = f.read_text(encoding="utf-8")
                        break
                    except Exception:
                        pass
        
        # Prepare JSON payload
        payload = {"code_snippet": code_snippet}
        
        if config.get("invalid_request"):
            # TC_19: Send malformed payload (missing code_snippet)
            payload = {"wrong_field": "test"}

        endpoint = config.get("endpoint", "/api/detect_smell_static")
        url = f"{gateway_url}{endpoint}"

        try:
            timeout = 10 if not config.get("timeout_test") else 0.001 
            
            # Use json=payload for application/json
            response = requests.post(url, json=payload, timeout=timeout)
            
            status = response.status_code
            
            if config["expected_status"] == "2xx":
                assert 200 <= status < 300, f"Expected 2xx, got {status}. Body: {response.text}"
                try:
                    data = response.json()
                    if isinstance(data, dict) and "success" in data:
                        assert data["success"] is True, f"Expected success=True, got {data}"
                except ValueError:
                    pass

            elif config["expected_status"] == "4xx":
                assert 400 <= status < 500, f"Expected 4xx, got {status}. Body: {response.text}"
            
            elif config["expected_status"] == "5xx":
                assert 500 <= status < 600, f"Expected 5xx, got {status}. Body: {response.text}"
            
            elif config["expected_status"] == "!=2xx":
                assert not (200 <= status < 300), f"Expected != 2xx, got {status}. Body: {response.text}"

            elif config["expected_status"] == "200_error_wrapped":
                # Gateway returns 200 but body implies failure
                assert status == 200, f"Expected 200 (wrapped error), got {status}"
                data = response.json()
                assert data.get("success") is False or "error" in data, f"Expected wrapped error, got {data}"

            elif config["expected_status"] == "200_validation_error":
                 # Gateway returns 200 but body contains 'detail' (FastAPI validation error)
                 assert status == 200, f"Expected 200 (wrapped validation), got {status}"
                 data = response.json()
                 assert "detail" in data, f"Expected validation error detail, got {data}"

            # Write result to output_dir
            with open(output_dir / "execution.log", "w") as f:
                f.write(f"Test: {tc_dir}\n")
                f.write(f"Status Code: {status}\n")
                f.write(f"Response Body: {response.text}\n")

        except requests.exceptions.ConnectionError:
            with open(output_dir / "execution.log", "w") as f:
                f.write(f"Test: {tc_dir}\n")
                f.write("Result: ConnectionError (Backend unreachable)\n")

            if config.get("expected_error") and config.get("timeout_test") == False:
                 # If we just can't connect to Gateway, that is a FAIL for system test environments unless specifically testing that.
                 # However, TC_18 is "Backend unreachable", not "Gateway unreachable". 
                 # If Gateway is unreachable, the test cannot proceed.
                 pass

            pytest.fail(f"Gateway at {url} not reachable. Environment is broken for {tc_dir}.")

        except requests.exceptions.Timeout:
            with open(output_dir / "execution.log", "w") as f:
                f.write(f"Test: {tc_dir}\n")
                f.write("Result: Timeout\n")

            if config.get("timeout_test"):
                # TC_20: Timeout is an acceptable outcome (if 504 not received but client timed out)
                return
            pytest.fail(f"Request timed out for {tc_dir}")

        except Exception as e:
            with open(output_dir / "execution.tlog", "w") as f:
                f.write(f"Test: {tc_dir}\n")
                f.write(f"Result: Unexpected Exception: {e}\n")

            if config.get("timeout_test"):
                 return
            pytest.fail(f"Unexpected exception for {tc_dir}: {e}")
            
        return

    # 2) Gestione permessi non leggibili (OUT2 o SP3)
    # Note: Step 1 (mkdir) is now done at the top
    locked_paths: List[tuple[Path, Optional[int]]] = []

    # Caso OUT2: output non accessibile
    if config.get("lock_output"):
        original_mode = make_unreadable(output_dir)
        locked_paths.append((output_dir, original_mode))

    # Caso SP3: sottocartelle inaccessibili
    for relative in config.get("unreadable_paths", []):
        target = tc_path / relative
        original_mode = make_unreadable(target)
        locked_paths.append((target, original_mode))

    # 3) Costruzione del comando CLI

    cmd: List[str] = [
        sys.executable,
        "-m",
        "cli.cli_runner",
        "--input",
        str(tc_path),
        "--output",
        str(output_dir),
    ]

    if config.get("parallel"):
        cmd.append("--parallel")
        cmd.extend(["--max_walkers", str(config.get("max_walkers", 5))])

    if config.get("resume"):
        cmd.append("--resume")

    if config.get("multiple"):
        cmd.append("--multiple")

    # Esecuzione dal root del repository (adatta se necessario)
    repo_root = Path(__file__).resolve().parents[2]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )

    # 4) Ripristino permessi
    for path_obj, mode in locked_paths:
        make_readable(path_obj, mode)

    # 5) Verifica oracolo: exit code
    if config["expected_error"]:
        assert completed.returncode != 0, (
            f"{tc_dir} doveva fallire, ma exit code = 0.\n"
            f"Stderr:\n{completed.stderr}"
        )
        # In caso di errore non ci aspettiamo report significativi
        return

    # Caso senza errore
    assert completed.returncode == 0, (
        f"{tc_dir} doveva completarsi con successo, ma exit code = {completed.returncode}.\n"
        f"Stderr:\n{completed.stderr}"
    )

    # 6) Verifica oracolo: numero di smell tramite overview.csv UNICO
    overview = output_dir / "overview.csv"
    expected_smells = config.get("expected_smells")

    # expected_smells = None => nessun report di smell previsto
    if expected_smells is None:
        if overview.exists():
            df = pd.read_csv(overview)
            assert df.empty, (
                f"{tc_dir}: non erano attesi report di smell, ma overview.csv contiene "
                f"{len(df)} righe."
            )
        return

    # expected_smells = 0 => o nessun file o file vuoto
    if not overview.exists():
        # Acceptable only if we ci aspettiamo 0 smell
        assert expected_smells == 0, (
            f"{tc_dir}: attesi {expected_smells} smell, ma overview.csv è assente."
        )
        return

    df = pd.read_csv(overview)
    smell_count = len(df)

    if expected_smells == 0:
        assert smell_count == 0, (
            f"{tc_dir}: attesi 0 smell, trovati {smell_count}."
        )
    elif expected_smells == ">=2":
        assert smell_count >= 2, (
            f"{tc_dir}: attesi almeno 2 smell, trovati {smell_count}."
        )
    else:
        # expected_smells è un intero > 0
        assert smell_count == expected_smells, (
            f"{tc_dir}: attesi {expected_smells} smell, trovati {smell_count}."
        )


