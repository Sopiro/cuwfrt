include(FetchContent)

# Include alzartak
FetchContent_Declare(
    alzartak
    GIT_REPOSITORY "https://github.com/Sopiro/alzartak.git"
    GIT_TAG master
)
FetchContent_MakeAvailable(alzartak)

file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
file(GLOB_RECURSE HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "src" FILES ${SOURCE_FILES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "include" FILES ${HEADER_FILES})

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

# set(CMAKE_CUDA_ARCHITECTURES 70 75 80) # Volta, Turing, Ampere

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO

    CUDA_ARCHITECTURES native
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED YES
)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:/WX /W4 /wd4819>
    )

    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=/wd4819
        -Wno-deprecated-gpu-targets
        >
    )
endif()

target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)