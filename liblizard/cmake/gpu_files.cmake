# ai: The GPU files the engines read (2026-10-03), out of the generated tree (out/: gen's indexes, SPIR-V and blobs):
# ai: each decoder variant's index (setup/<variant>.json, which core/dec/build.cpp makes the variant's setup through
# ai: since 2026-11-25), the modules it names (spv/<name>.spv) and the nets' files it names (blobs/<hash>.bin), the
# ai: camera's ingest kernel, the sender's index (setup/send.json, each module's recipe: core/tx/gpu_painter.cpp holds it
# ai: to its own; its tables the C's since 2026-11-25, no blob) and its send_*.spv. Nothing else of the tree: its raw SPIR-V, the record of the JS setup
# ai: (record/) and the benchmark's kernel are not read. The one copy of the rule: the library's install takes it by the
# ai: function, and a build that is not CMake's (the AAR, bindings/kotlin/lizard-android) runs this file as a script for
# ai: the list.
# ai:   liz_gpu_files(<out var> <tree> <variants...>): the files, relative to the tree
# ai:   cmake -DLIZ_GPU_TREE=<tree> "-DLIZ_GPU_VARIANTS=<v;v...>" -DLIZ_GPU_LIST=<file> -P gpu_files.cmake: the files
# ai:     written to <file>, one a line
function(liz_gpu_files out tree)
  set(files setup/send.json spv/ingest_camera.spv)
  file(GLOB sends RELATIVE ${tree} ${tree}/spv/send_*.spv)
  list(FILTER sends EXCLUDE REGEX "\\.raw\\.spv$")
  list(APPEND files ${sends})
  foreach(v ${ARGN})
    if(NOT EXISTS ${tree}/setup/${v}.json)
      message(FATAL_ERROR "no GPU setup ${v} in ${tree}")
    endif()
    list(APPEND files setup/${v}.json)
    file(READ ${tree}/setup/${v}.json j)
    string(JSON mods GET "${j}" modules)
    string(JSON nm LENGTH "${mods}")
    math(EXPR last "${nm} - 1")
    foreach(i RANGE ${last})
      string(JSON m GET "${mods}" ${i} name)
      list(APPEND files spv/${m}.spv)
    endforeach()
    string(JSON nets GET "${j}" nets)
    string(JSON nn LENGTH "${nets}")
    math(EXPR last "${nn} - 1")
    foreach(i RANGE ${last})
      string(JSON f MEMBER "${nets}" ${i})
      string(JSON b GET "${nets}" ${f})
      list(APPEND files blobs/${b}.bin)
    endforeach()
  endforeach()
  list(REMOVE_DUPLICATES files)
  set(${out} ${files} PARENT_SCOPE)
endfunction()

if(CMAKE_SCRIPT_MODE_FILE STREQUAL CMAKE_CURRENT_LIST_FILE)
  if(NOT LIZ_GPU_TREE OR NOT LIZ_GPU_VARIANTS OR NOT LIZ_GPU_LIST)
    message(FATAL_ERROR "cmake -DLIZ_GPU_TREE=<tree> -DLIZ_GPU_VARIANTS=<v;v...> -DLIZ_GPU_LIST=<file> -P gpu_files.cmake")
  endif()
  if(NOT EXISTS ${LIZ_GPU_TREE}/setup/send.json)
    message(FATAL_ERROR "no generated GPU tree in ${LIZ_GPU_TREE}: lizard-android/build.sh tools, then gen")
  endif()
  liz_gpu_files(files ${LIZ_GPU_TREE} ${LIZ_GPU_VARIANTS})
  list(JOIN files "\n" text)
  file(WRITE ${LIZ_GPU_LIST} "${text}\n")
endif()
