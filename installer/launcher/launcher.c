/* Star Effects Lab launcher: embeds the bundled Python and runs `python -m starlab <args>`.
 * macOS: Contents/MacOS/<this exe>, Python in Contents/Resources/python.
 * Windows: <root>\Star Effects Lab.exe (GUI subsystem), Python in <root>\python. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <libgen.h>
#include <mach-o/dyld.h>

/* Writes <bundle>/Contents/Resources/python into out; returns 0 on success. */
static int python_home(char *out, size_t cap) {
    char raw[PATH_MAX], real[PATH_MAX];
    uint32_t n = sizeof raw;
    if (_NSGetExecutablePath(raw, &n) != 0 || !realpath(raw, real)) return -1;
    char *macos = dirname(real);               /* .../Contents/MacOS */
    char *contents = dirname(macos);           /* .../Contents */
    return snprintf(out, cap, "%s/Resources/python", contents) < (int)cap ? 0 : -1;
}

int main(int argc, char **argv) {
    char home[PATH_MAX], exe[PATH_MAX + 16];
    if (python_home(home, sizeof home) != 0) {
        fprintf(stderr, "Star Effects Lab: cannot locate bundled Python\n");
        return 1;
    }
    snprintf(exe, sizeof exe, "%s/bin/python3", home);

    /* argv for Python: <name> -m starlab <forwarded args> */
    char **pyargv = calloc((size_t)argc + 3, sizeof *pyargv);
    if (!pyargv) return 1;
    pyargv[0] = "Star Effects Lab";
    pyargv[1] = "-m";
    pyargv[2] = "starlab";
    for (int i = 1; i < argc; i++) pyargv[i + 2] = argv[i];

    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.use_environment = 0;        /* ignore PYTHONPATH / PYTHONHOME */
    config.user_site_directory = 0;
    config.safe_path = 1;              /* keep the working directory off sys.path (stray .py files shadow stdlib) */
    config.install_signal_handlers = 1;

    PyStatus st = PyConfig_SetBytesString(&config, &config.home, home);
    if (!PyStatus_Exception(st)) st = PyConfig_SetBytesString(&config, &config.executable, exe);
    if (!PyStatus_Exception(st)) st = PyConfig_SetBytesString(&config, &config.program_name, exe);
    if (!PyStatus_Exception(st)) st = PyConfig_SetBytesArgv(&config, argc + 2, pyargv);
    if (!PyStatus_Exception(st)) st = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(st)) {
        if (PyStatus_IsExit(st)) return st.exitcode;
        Py_ExitStatusException(st);
    }
    return Py_RunMain();
}

#elif defined(_WIN32)
#include <windows.h>
#include <shellapi.h>

/* Windows: <root>\Star Effects Lab.exe, Python in <root>\python. python314.dll is delay-loaded (see
 * CMakeLists) and pre-loaded here from that folder so it and its vcruntime dependency are found. */
int WINAPI wWinMain(HINSTANCE inst, HINSTANCE prev, PWSTR cmd, int show) {
    (void)inst; (void)prev; (void)cmd; (void)show;
    wchar_t self[MAX_PATH], home[MAX_PATH + 16], exe[MAX_PATH + 32], dll[MAX_PATH + 48];
    DWORD n = GetModuleFileNameW(NULL, self, MAX_PATH);
    wchar_t *slash = wcsrchr(self, L'\\');
    if (n == 0 || n >= MAX_PATH || !slash) return 1;
    *slash = 0;
    swprintf(home, sizeof home / sizeof *home, L"%ls\\python", self);
    swprintf(exe, sizeof exe / sizeof *exe, L"%ls\\python.exe", home);
    swprintf(dll, sizeof dll / sizeof *dll, L"%ls\\" PY_DLL_NAME, home);
    if (!LoadLibraryExW(dll, NULL, LOAD_WITH_ALTERED_SEARCH_PATH)) {
        MessageBoxW(NULL, L"Cannot load the bundled Python. Reinstall Star Effects Lab.",
                    L"Star Effects Lab", MB_OK | MB_ICONERROR);
        return 1;
    }

    int argc = 0;
    wchar_t **argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argv) return 1;
    wchar_t **pyargv = calloc((size_t)argc + 3, sizeof *pyargv);
    if (!pyargv) return 1;
    pyargv[0] = L"Star Effects Lab";
    pyargv[1] = L"-m";
    pyargv[2] = L"starlab";
    for (int i = 1; i < argc; i++) pyargv[i + 2] = argv[i];

    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.use_environment = 0;        /* ignore PYTHONPATH / PYTHONHOME */
    config.user_site_directory = 0;
    config.safe_path = 1;              /* keep the working directory off sys.path (stray .py files shadow stdlib) */
    config.install_signal_handlers = 1;

    PyStatus st = PyConfig_SetString(&config, &config.home, home);
    if (!PyStatus_Exception(st)) st = PyConfig_SetString(&config, &config.executable, exe);
    if (!PyStatus_Exception(st)) st = PyConfig_SetString(&config, &config.program_name, exe);
    if (!PyStatus_Exception(st)) st = PyConfig_SetArgv(&config, argc + 2, pyargv);
    if (!PyStatus_Exception(st)) st = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);
    if (PyStatus_Exception(st)) {
        if (PyStatus_IsExit(st)) return st.exitcode;
        wchar_t msg[600] = L"Star Effects Lab could not start its Python runtime.\n\n";
        size_t used = wcslen(msg);
        if (st.err_msg) MultiByteToWideChar(CP_UTF8, 0, st.err_msg, -1, msg + used, (int)(600 - used));
        MessageBoxW(NULL, msg, L"Star Effects Lab", MB_OK | MB_ICONERROR);
        return 1;
    }
    return Py_RunMain();
}

#else
#error "Unsupported platform"
#endif
