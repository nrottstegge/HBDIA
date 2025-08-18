# CMake generated Testfile for 
# Source directory: /users/nrottstegge/github/HBDIA/tests
# Build directory: /users/nrottstegge/github/HBDIA/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[HBDIA_conversion_test]=] "/users/nrottstegge/github/HBDIA/build/tests/test_print_HBDIA")
set_tests_properties([=[HBDIA_conversion_test]=] PROPERTIES  _BACKTRACE_TRIPLES "/users/nrottstegge/github/HBDIA/tests/CMakeLists.txt;42;add_test;/users/nrottstegge/github/HBDIA/tests/CMakeLists.txt;0;")
add_test([=[HBDIA_cusparse_comparison_test]=] "mpirun" "-n" "2" "test_HBDIA_cusparse_comparison")
set_tests_properties([=[HBDIA_cusparse_comparison_test]=] PROPERTIES  _BACKTRACE_TRIPLES "/users/nrottstegge/github/HBDIA/tests/CMakeLists.txt;43;add_test;/users/nrottstegge/github/HBDIA/tests/CMakeLists.txt;0;")
