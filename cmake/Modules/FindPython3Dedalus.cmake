# FindPython3Dedalus.cmake
# 
# Finds Python3 with NumPy for the Dedalus interface.
# Uses CMake's built-in FindPython3 but provides channelflow-specific setup.
#
# This module sets the following variables:
#   PYTHON3_DEDALUS_FOUND       - True if Python3 and NumPy were found
#   PYTHON3_DEDALUS_INCLUDE_DIRS - Include directories for Python and NumPy
#   PYTHON3_DEDALUS_LIBRARIES   - Libraries to link against
#   PYTHON3_DEDALUS_VERSION     - Python version string
#
# This module defines the following imported targets:
#   Python3Dedalus::Python      - The Python library target
#

# Use CMake's built-in Python3 finder
find_package(Python3 COMPONENTS Interpreter Development NumPy QUIET)

if(Python3_FOUND AND Python3_NumPy_FOUND)
    set(PYTHON3_DEDALUS_FOUND TRUE)
    set(PYTHON3_DEDALUS_INCLUDE_DIRS ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
    set(PYTHON3_DEDALUS_LIBRARIES Python3::Python)
    set(PYTHON3_DEDALUS_VERSION ${Python3_VERSION})
    
    # Create an interface target (ALIAS of imported targets requires GLOBAL visibility)
    if(NOT TARGET Python3Dedalus::Python)
        add_library(Python3Dedalus::Python INTERFACE IMPORTED GLOBAL)
        target_link_libraries(Python3Dedalus::Python INTERFACE Python3::Python)
    endif()
    
    message(STATUS "Python3 found: ${Python3_EXECUTABLE} (version ${Python3_VERSION})")
    message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
    message(STATUS "Python3 NumPy include: ${Python3_NumPy_INCLUDE_DIRS}")
    message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")
else()
    set(PYTHON3_DEDALUS_FOUND FALSE)
    if(Python3Dedalus_FIND_REQUIRED)
        message(FATAL_ERROR "Python3 with NumPy required for Dedalus interface, but not found.")
    else()
        message(STATUS "Python3 with NumPy not found.")
    endif()
endif()

mark_as_advanced(
    PYTHON3_DEDALUS_INCLUDE_DIRS
    PYTHON3_DEDALUS_LIBRARIES
)
