#include "slang-cpp-prelude.h"

#ifdef SLANG_PRELUDE_NAMESPACE
using namespace SLANG_PRELUDE_NAMESPACE;
#endif


#line 1 "/tmp/emitted_shaders/axpy.slang"
struct AxpyParams_0
{
    uint32_t n_0;
    float alpha_0;
};


#line 19
struct GlobalParams_0
{
    AxpyParams_0* params_0;
    StructuredBuffer<float> x_0;
    RWStructuredBuffer<float> y_0;
};


#line 19
struct KernelContext_0
{
    GlobalParams_0* globalParams_0;
};


#line 14
void _main_0(void* _S1, void* entryPointParams_0, void* globalParams_1)
{

#line 14
    ComputeThreadVaryingInput * _S2 = (slang_bit_cast<ComputeThreadVaryingInput *>(_S1));

#line 14
    KernelContext_0 kernelContext_0;

#line 14
    (&kernelContext_0)->globalParams_0 = (slang_bit_cast<GlobalParams_0*>(globalParams_1));
    uint32_t i_0 = (_S2->groupID * Vector<uint32_t, 3> (256U, 1U, 1U) + _S2->groupThreadID).x;
    if(i_0 >= ((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->n_0))
    {

#line 17
        return;
    }
    *(&((&kernelContext_0)->globalParams_0->y_0)[i_0]) = (F32_fma(((slang_bit_cast<GlobalParams_0*>(globalParams_1))->params_0->alpha_0), ((&kernelContext_0)->globalParams_0->x_0.Load(i_0)), (*(&((&kernelContext_0)->globalParams_0->y_0)[i_0]))));
    return;
}

// [numthreads(256, 1, 1)]
SLANG_PRELUDE_EXPORT
void main_0_Thread(ComputeThreadVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    _main_0(varyingInput, entryPointParams, globalParams);
}
// [numthreads(256, 1, 1)]
SLANG_PRELUDE_EXPORT
void main_0_Group(ComputeVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    ComputeThreadVaryingInput threadInput = {};
    threadInput.groupID = varyingInput->startGroupID;
    for (uint32_t x = 0; x < 256; ++x)
    {
        threadInput.groupThreadID.x = x;
        _main_0(&threadInput, entryPointParams, globalParams);
    }
}
// [numthreads(256, 1, 1)]
SLANG_PRELUDE_EXPORT
void main_0(ComputeVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    ComputeVaryingInput vi = *varyingInput;
    ComputeVaryingInput groupVaryingInput = {};
    for (uint32_t z = vi.startGroupID.z; z < vi.endGroupID.z; ++z)
    {
        groupVaryingInput.startGroupID.z = z;
        for (uint32_t y = vi.startGroupID.y; y < vi.endGroupID.y; ++y)
        {
            groupVaryingInput.startGroupID.y = y;
            for (uint32_t x = vi.startGroupID.x; x < vi.endGroupID.x; ++x)
            {
                groupVaryingInput.startGroupID.x = x;
                main_0_Group(&groupVaryingInput, entryPointParams, globalParams);
            }
        }
    }
}
