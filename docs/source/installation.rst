Installation
============

Prerequisites
-------------

- Python 3.8 or higher
- Git (for cloning repository)

Platform Compatibility
----------------------

muNet-AUVsim runs best on Linux. The included AquaNet communication stack is not
compatible with Windows, but use of AquaNet is optional. The core simulation
framework supports Windows using the muNet network simulator or direct-access.

*Use of AquaNet is not required to run muNet-AUVsim.* It is only one of the
available options for simulating a communication network.

**Core Simulation Framework:**

- **Linux** - Full support (tested)
- **macOS** - Expected to work (not extensively tested)
- **Windows** - Core features supported with muNet communication

**AquaNet Communication Stack:**

- **Linux** - AquaNet requires Unix domain sockets and Linux binaries
- **macOS** - May work but untested
- **Windows** - Not supported: use muNet instead

Windows Users
~~~~~~~~~~~~~

Windows users can run the complete simulation framework with the following
limitation:

**Available:**

- All vehicle dynamics and guidance, navigation, and control features
- muNet acoustic network simulator (full-featured simulation)
- Ocean environment modeling
- Visualization and data collection
- Simulation saving and loading

**Not Available:**

- AquaNet protocol stack integration
- Use ``loadMuNet()`` instead of ``loadAquaNet()`` in your simulations

Step-by-Step Installation
--------------------------

1. **Clone the repository:**

   .. code-block:: bash

      git clone https://github.com/munet-ua/munet-auvsim.git
      cd munet-auvsim

2. **Create a Python virtual environment:**

   .. code-block:: bash

      # Choose your preferred virtual environment name (e.g. 'munet')
      python -m venv munet

   This creates a directory for the virtual environment named 'munet' in your
   project folder.

3. **Add the virtual environment folder to your git exclude list:**

   .. code-block:: bash

      # Open in your preferred text editor
      gedit .git/info/exclude

   Add the virtual environment directory name on a new line and save:

   .. code-block:: text

      munet/

4. **Add the project to the PYTHONPATH for your virtual environment:**

   **Linux/Mac:**

   .. code-block:: bash

      # Open virtual environment activation script
      gedit munet/bin/activate

   Add at the end of the file (replace with your actual path):

   .. code-block:: bash

      export PYTHONPATH="/path/to/your/project/munet-auvsim"

   **Windows:**

   .. code-block:: batch

      # Open virtual environment activation script
      notepad munet\Scripts\activate.bat

   Add before the final line:

   .. code-block:: batch

      set PYTHONPATH=C:\path\to\munet-auvsim;%PYTHONPATH%

5. **Activate the python virtual environment:**

   **Linux/Mac:**

   .. code-block:: bash

      source munet/bin/activate

   **Windows:**

   .. code-block:: batch

      munet\Scripts\activate

   You should see that your prompt is now updated with ``(munet)`` to indicate
   you are working inside a python virtual environment (venv). To exit the venv,
   type ``deactivate``.

6. **Install the required dependencies:**

   Make sure you are in the virtual environment (e.g. '(munet)' shows at
   prompt). Enter:

   .. code-block:: bash

      pip install -r requirements.txt

Verifying Installation
----------------------

Test your installation:

.. code-block:: python

   python -c "import munetauvsim as mn; print(f'muNet-AUVsim v{mn.__version__} loaded successfully')"


Verify everything works by running the quick-start example:

.. code-block:: bash

    cd munet-auvsim
    python scripts/example.py

Expected output:

- Simulation runs for ~60 seconds
- Simulation progress printed to console
- Output files created in ``outputs/example/Example_YYMMDD-HHMMSS`` directory:

  - Simulation log file
  - 3D trajectory animation (GIF)
  - Simulation data file (pickle)

If the script runs without errors, your installation is verified and ready to use.

For a more detailed interactive experience, try the tutorial script:

.. code-block:: bash

    python scripts/demo.py

**Windows users:** If you see an error related to ``aquanet_lib``, this is
expected. AquaNet is Linux-only. The core simulation framework will work
correctly with muNet communication.