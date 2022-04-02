#pragma once
#include "utils.h"
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/functional.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

template <typename ElementOutput_,
          typename ElementAccumulator_ = ElementOutput_, bool Source_ = false>
class Dequant {
  public:
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementOutput_;

    static int const kCount = 1;

    using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;

    /// Host-constructable parameters structure
    struct Params {

        ElementOutput scaleA;            ///< scales accumulators
        ElementOutput scaleB;            ///< scales source tensor
        ElementOutput const *scaleA_ptr; ///< pointer to accumulator scalar -
                                         ///< if not null, loads it from memory
        ElementOutput const *scaleB_ptr; ///< pointer to source scalar - if not
                                         ///< null, loads it from memory
        ElementOutput *scaleC_ptr;

        CUTLASS_HOST_DEVICE
        Params()
            : scaleA(ElementOutput(1)), scaleB(ElementOutput(1)),
              scaleA_ptr(nullptr), scaleB_ptr(nullptr), scaleC_ptr(nullptr) {}

        CUTLASS_HOST_DEVICE
        Params(ElementOutput *scaleC_ptr)
            : scaleA(ElementOutput(1)), scaleB(ElementOutput(1)),
              scaleA_ptr(nullptr), scaleB_ptr(nullptr), scaleC_ptr(scaleC_ptr) {
        }

        CUTLASS_HOST_DEVICE
        Params(ElementOutput scaleA, ElementOutput scaleB,
               ElementOutput *scaleC_ptr)
            : scaleA(scaleA), scaleB(scaleB), scaleA_ptr(nullptr),
              scaleB_ptr(nullptr), scaleC_ptr(scaleC_ptr) {}

        CUTLASS_HOST_DEVICE
        Params(ElementOutput const *scaleA_ptr, ElementOutput const *scaleB_ptr,
               ElementOutput *scaleC_ptr)
            : scaleA(1), scaleB(1), scaleA_ptr(scaleA_ptr),
              scaleB_ptr(scaleB_ptr), scaleC_ptr(scaleC_ptr) {}
    };

  public:
    ElementOutput scale;
    ElementOutput *scaleC_ptr;
    CUTLASS_HOST_DEVICE
    Dequant(Params const &params) {
        // auto scaleA_ = (params.scaleA_ptr ? *params.scaleA_ptr :
        // params.scaleA); auto scaleB_ = (params.scaleB_ptr ?
        // *params.scaleB_ptr : params.scaleB);
        auto scaleA_ =
            (params.scaleA_ptr ? __ldg(params.scaleA_ptr) : params.scaleA);
        auto scaleB_ =
            (params.scaleB_ptr ? __ldg(params.scaleB_ptr) : params.scaleB);

        scale = scaleA_ * scaleB_;
        scaleC_ptr = params.scaleC_ptr;
    }

    /// Returns true if source is needed
    CUTLASS_HOST_DEVICE
    bool is_source_needed() const { return Source_; }

    /// Functionally required for serial reduction in the epilogue
    CUTLASS_HOST_DEVICE
    void set_k_partition(int k_partition, int k_partition_count) {}

    /// Dequantize the result from int32_t to float
    CUTLASS_HOST_DEVICE FragmentOutput
    operator()(FragmentAccumulator const &accumulator) const {
        cutlass::NumericArrayConverter<ElementOutput, ElementAccumulator,
                                       kCount>
            accumulator_converter;
        auto converted_accumulator = accumulator_converter(accumulator);
        cutlass::multiplies<FragmentOutput> mul_accumulator;
        return mul_accumulator(scale, converted_accumulator);
    }

    /// Add C after dequantization
    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(FragmentAccumulator const &accumulator,
                              FragmentOutput const &source) const {
        cutlass::plus<FragmentOutput> plus_accumulator;
        return plus_accumulator(source, operator()(accumulator));
    }
};

#include <cutlass/epilogue/threadblock/epilogue_base.h>

template <typename Frag_t>
CUTLASS_HOST_DEVICE void fragment_max(float &result, Frag_t const &frag) {
#pragma unroll
    for (int i = 0; i < Frag_t::kElements; ++i) {
        result = max(result, fabs(frag[i]));
    }
}

template <typename Shape_, typename WarpMmaOperator_, int PartitionsK,
          typename OutputTileIterator_, typename AccumulatorFragmentIterator_,
          typename WarpTileIterator_, typename SharedLoadIterator_,
          typename OutputOp_, typename Padding_, int FragmentsPerPartition = 1,
          int IterationsUnroll = false>
class EpilogueReduction
    : public cutlass::epilogue::threadblock::EpilogueBase<
          Shape_, typename WarpMmaOperator_::Shape, PartitionsK,
          AccumulatorFragmentIterator_, WarpTileIterator_, Padding_,
          FragmentsPerPartition> {

  public:
    using Base = typename cutlass::epilogue::threadblock::EpilogueBase<
        Shape_, typename WarpMmaOperator_::Shape, PartitionsK,
        AccumulatorFragmentIterator_, WarpTileIterator_, Padding_,
        FragmentsPerPartition>;

    using Shape = Shape_;
    using WarpMmaOperator = WarpMmaOperator_;
    static int const kPartitionsK = PartitionsK;
    using OutputTileIterator = OutputTileIterator_;
    using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
    using WarpTileIterator = WarpTileIterator_;
    using SharedLoadIterator = SharedLoadIterator_;
    using OutputOp = OutputOp_;
    using Padding = Padding_;

    using Layout = cutlass::layout::RowMajor;
    using LongIndex = typename Layout::LongIndex;

    /// The complete warp-level accumulator tile
    using AccumulatorTile = typename Base::AccumulatorTile;

    /// Accumulator element
    using ElementAccumulator = typename WarpTileIterator::Element;

    /// Output element
    using ElementOutput = typename OutputTileIterator::Element;

    /// Output access size
    static int const kElementsPerAccess =
        OutputTileIterator::kElementsPerAccess;

    /// Tensor reference to destination tensor
    using TensorRef = typename OutputTileIterator::TensorRef;

    /// Tensor reference to sync tensor
    using SyncTensorRef =
        typename cutlass::TensorRef<int, cutlass::layout::PackedVectorLayout>;

    /// Const tensor reference to source tensor
    using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

    /// Array type used to output
    using OutputAccessType =
        cutlass::Array<typename OutputTileIterator::Element,
                       OutputTileIterator::kElementsPerAccess>;

    /// Array type used by output functor
    using AccumulatorAccessType =
        cutlass::Array<typename WarpTileIterator::Element,
                       OutputTileIterator::kElementsPerAccess>;

    /// Number of warps
    using WarpCount = typename Base::WarpCount;

    static int constexpr kSmemTiles = Base::kFragmentsPerIteration > 1
                                          ? Base::kFragmentsPerIteration
                                          : kPartitionsK;
    static int constexpr kSmemPointerOffset =
        Base::SharedStorage::StorageShape::kCount / kSmemTiles;

  public:
    static_assert(
        SharedLoadIterator::Fragment::kElements ==
            OutputTileIterator::Fragment::kElements,
        "Mismatch between shared load iterator and output tile iterator.");

    static_assert(OutputTileIterator::kElementsPerAccess,
                  "OutputTileIterator::kElementsPerAccess must not be zero.");

    static_assert(!(OutputTileIterator::Fragment::kElements %
                    OutputTileIterator::kElementsPerAccess),
                  "Divisibility");

  private:
    /// Loads fragment from shared memory aligned with output tensor
    SharedLoadIterator shared_load_iterator_;

  public:
    /// Constructor
    CUTLASS_DEVICE
    EpilogueReduction(
        typename Base::SharedStorage &shared_storage, ///< Shared storage object
        int thread_idx, ///< ID of a thread within the threadblock
        int warp_idx,   ///< ID of warp within threadblock
        int lane_idx    ///< Id of thread within warp
        )
        : Base(shared_storage, thread_idx, warp_idx, lane_idx),
          shared_load_iterator_(shared_storage.reference(), thread_idx) {}

    /// Streams the result to global memory
    CUTLASS_DEVICE
    void
    operator()(OutputOp const &output_op, ///< Output operator
               OutputTileIterator
                   destination_iterator, ///< Tile iterator for destination
               AccumulatorTile const
                   &accumulators, ///< Complete warp-level accumulator tile
               OutputTileIterator
                   source_iterator) { ///< Threadblock tile coordinate in GEMM
                                      ///< (in units of threadblock tiles)

        if (!output_op.is_source_needed()) {
            compute_source_not_needed_(output_op, destination_iterator,
                                       accumulators);
        } else {
            compute_source_needed_(output_op, destination_iterator,
                                   accumulators, source_iterator);
        }
    }

  private:
    template <class Seq> struct acc2smem_source_not_needed;

    template <size_t... Seq>
    struct acc2smem_source_not_needed<cutlass::index_sequence<Seq...>> {
        template <int Advance>
        CUTLASS_DEVICE static void
        helper(AccumulatorFragmentIterator accum_fragment_iterator,
               WarpTileIterator &warp_tile_iterator) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < Advance; i++) {
                ++accum_fragment_iterator;
            }

            CUTLASS_PRAGMA_UNROLL
            for (int p = 0; p < Base::kFragmentsPerIteration; ++p) {
                typename AccumulatorFragmentIterator::Fragment accum_fragment;

                accum_fragment_iterator.load(accum_fragment);
                ++accum_fragment_iterator;

                warp_tile_iterator.store(accum_fragment);
                if (p < Base::kFragmentsPerIteration - 1) {
                    warp_tile_iterator.add_pointer_offset(kSmemPointerOffset);
                }
            }

            if (Base::kFragmentsPerIteration > 1) {
                warp_tile_iterator.add_pointer_offset(
                    kSmemPointerOffset * (1 - Base::kFragmentsPerIteration));
            }
        }

        CUTLASS_DEVICE
        static void push(size_t pos,
                         AccumulatorFragmentIterator const &iterator_begin,
                         WarpTileIterator &warp_tile_iterator) {
            int dummy[] = {(pos == (Seq * Base::kFragmentsPerIteration)) &&
                           (helper<Seq * Base::kFragmentsPerIteration>(
                                iterator_begin, warp_tile_iterator),
                            0)...};

            CUTLASS_UNUSED(dummy[0]);
        }
    };

    static_assert(kPartitionsK == 1 || Base::kFragmentsPerIteration == 1,
                  "One of these must be exactly 1.");

    /// Streams the result to global memory
    CUTLASS_DEVICE
    void compute_source_not_needed_(
        OutputOp const &output_op, ///< Output operator
        OutputTileIterator
            destination_iterator, ///< Tile iterator for destination
        AccumulatorTile const
            &accumulators ///< Complete warp-level accumulator tile
    ) {

        typename OutputOp::ElementOutput reduction_value = 0;

        //
        // Iterator over warp-level accumulator fragment
        //

        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        //
        // Iterate over accumulator tile
        //

#pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations /            \
                                      Base::kFragmentsPerIteration             \
                                : 1)
        for (int iter = 0; iter < OutputTileIterator::kIterations;
             iter += Base::kFragmentsPerIteration) {

            //
            // Convert and store fragment
            //

            __syncthreads();

            acc2smem_source_not_needed<cutlass::make_index_sequence<
                OutputTileIterator::kIterations /
                Base::kFragmentsPerIteration>>::push(iter,
                                                     accum_fragment_iterator,
                                                     this->warp_tile_iterator_);

            __syncthreads();

            //
            // Load fragments from shared memory
            //

            CUTLASS_PRAGMA_UNROLL
            for (int p = 0; p < Base::kFragmentsPerIteration; ++p) {

                typename SharedLoadIterator::Fragment
                    aligned_accum_fragment[kPartitionsK];

                shared_load_iterator_.load(aligned_accum_fragment[0]);

                if (p < Base::kFragmentsPerIteration - 1) {
                    shared_load_iterator_.add_pointer_offset(
                        kSmemPointerOffset);
                } else if (kPartitionsK > 1) {

                    cutlass::plus<typename SharedLoadIterator::Fragment>
                        add_fragments;

                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 1; i < kPartitionsK; ++i) {
                        shared_load_iterator_.add_pointer_offset(
                            kSmemPointerOffset);
                        shared_load_iterator_.load(aligned_accum_fragment[i]);
                        aligned_accum_fragment[0] =
                            add_fragments(aligned_accum_fragment[0],
                                          aligned_accum_fragment[i]);
                    }

                    shared_load_iterator_.add_pointer_offset(
                        (1 - kPartitionsK) * kSmemPointerOffset);
                }

                //
                // Compute the output result
                //

                typename OutputTileIterator::Fragment output_fragment;

                apply_output_operator_source_not_needed_(
                    output_fragment, output_op, aligned_accum_fragment[0]);

                // Reduce output fragment
                fragment_max(reduction_value, output_fragment);
                // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
                //     for (int i=0; i < decltype(output_fragment)::kElements;
                //     i++) {
                //         printf("%f ", output_fragment[i]);
                //     }
                //     printf("\n");
                // }
                //
                // Store the final result
                //
                destination_iterator.store(output_fragment);
                ++destination_iterator;
            }

            if (Base::kFragmentsPerIteration > 1) {
                shared_load_iterator_.add_pointer_offset(
                    kSmemPointerOffset * (1 - Base::kFragmentsPerIteration));
            }
        }

        // reduce
        scale_reduce(reduction_value / 127.0f, output_op.scaleC_ptr);
    }

    template <class Seq> struct acc2smem_source_needed;

    template <size_t... Seq>
    struct acc2smem_source_needed<cutlass::index_sequence<Seq...>> {
        template <int Advance>
        CUTLASS_DEVICE static void
        helper(AccumulatorFragmentIterator accum_fragment_iterator,
               WarpTileIterator &warp_tile_iterator) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < Advance; i++) {
                ++accum_fragment_iterator;
            }

            typename AccumulatorFragmentIterator::Fragment accum_fragment;
            accum_fragment_iterator.load(accum_fragment);
            warp_tile_iterator.store(accum_fragment);
        }

        CUTLASS_DEVICE
        static void push(size_t pos,
                         AccumulatorFragmentIterator const &iterator_begin,
                         WarpTileIterator &warp_tile_iterator) {
            int dummy[] = {
                (pos == Seq) &&
                (helper<Seq>(iterator_begin, warp_tile_iterator), 0)...};
        }
    };

    /// Streams the result to global memory
    CUTLASS_DEVICE
    void compute_source_needed_(
        OutputOp const &output_op, ///< Output operator
        OutputTileIterator
            destination_iterator, ///< Tile iterator for destination
        AccumulatorTile const
            &accumulators, ///< Complete warp-level accumulator tile
        OutputTileIterator
            source_iterator ///< Threadblock tile coordinate in GEMM (in units
                            ///< of threadblock tiles)
    ) {
        typename OutputOp::ElementOutput reduction_value = 0;

        typename OutputTileIterator::Fragment source_fragment;

        source_fragment.clear();

        //
        // Iterator over warp-level accumulator fragment
        //

        AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

        //
        // Iterate over accumulator tile
        //

#pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
        for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {

            //
            // Load the source
            //

            source_iterator.load(source_fragment);
            ++source_iterator;

            //
            // Convert and store fragment
            //

            __syncthreads();

            acc2smem_source_needed<
                cutlass::make_index_sequence<OutputTileIterator::kIterations>>::
                push(iter, accum_fragment_iterator, this->warp_tile_iterator_);

            __syncthreads();

            //
            // Load fragments from shared memory
            //

            typename SharedLoadIterator::Fragment
                aligned_accum_fragment[kPartitionsK];

            shared_load_iterator_.load(aligned_accum_fragment[0]);

            // If the number of k-slices is > 1 - perform a reduction amongst
            // the k-slices
            if (kPartitionsK > 1) {

                cutlass::plus<typename SharedLoadIterator::Fragment>
                    add_fragments;

                CUTLASS_PRAGMA_UNROLL
                for (int i = 1; i < kPartitionsK; ++i) {
                    shared_load_iterator_.add_pointer_offset(
                        kSmemPointerOffset);
                    shared_load_iterator_.load(aligned_accum_fragment[i]);
                    aligned_accum_fragment[0] = add_fragments(
                        aligned_accum_fragment[0], aligned_accum_fragment[i]);
                }

                shared_load_iterator_.add_pointer_offset((1 - kPartitionsK) *
                                                         kSmemPointerOffset);
            }

            //
            // Compute the output result
            //

            typename OutputTileIterator::Fragment output_fragment;

            apply_output_operator_(output_fragment, output_op,
                                   aligned_accum_fragment[0], source_fragment);

            // Reduce output fragment
            fragment_max(reduction_value, output_fragment);
            //
            // Store the final result
            //
            destination_iterator.store(output_fragment);
            ++destination_iterator;
        }

        // reduce
        scale_reduce(reduction_value / 127.0f, output_op.scaleC_ptr);
    }

    /// Helper to invoke the output functor over each vector of output
    CUTLASS_DEVICE
    void apply_output_operator_(
        typename OutputTileIterator::Fragment &output_fragment,
        OutputOp const &output_op, ///< Output operator
        typename SharedLoadIterator::Fragment const &aligned_accum_fragment,
        typename OutputTileIterator::Fragment const &source_fragment) {

        OutputAccessType *output_frag_ptr =
            reinterpret_cast<OutputAccessType *>(&output_fragment);

        AccumulatorAccessType const *compute_frag_ptr =
            reinterpret_cast<AccumulatorAccessType const *>(
                &aligned_accum_fragment);

        OutputAccessType const *source_frag_ptr =
            reinterpret_cast<OutputAccessType const *>(&source_fragment);

        int const kOutputOpIterations =
            OutputTileIterator::Fragment::kElements /
            OutputTileIterator::kElementsPerAccess;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kOutputOpIterations; ++i) {

            // Call the output operator
            output_frag_ptr[i] =
                output_op(compute_frag_ptr[i], source_frag_ptr[i]);
        }
    }

    /// Helper to invoke the output functor over each vector of output
    CUTLASS_DEVICE
    void apply_output_operator_source_not_needed_(
        typename OutputTileIterator::Fragment &output_fragment,
        OutputOp const &output_op, ///< Output operator
        typename SharedLoadIterator::Fragment const &aligned_accum_fragment) {

        OutputAccessType *output_frag_ptr =
            reinterpret_cast<OutputAccessType *>(&output_fragment);

        AccumulatorAccessType const *compute_frag_ptr =
            reinterpret_cast<AccumulatorAccessType const *>(
                &aligned_accum_fragment);

        int const kOutputOpIterations =
            OutputTileIterator::Fragment::kElements /
            OutputTileIterator::kElementsPerAccess;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kOutputOpIterations; ++i) {

            // Call the output operator
            output_frag_ptr[i] = output_op(compute_frag_ptr[i]);
        }
    }
};

template <typename Shape_, typename WarpMmaSimt_, typename OutputOp_,
          int ElementsPerAccess>
struct EpilogueHelper {
    using Shape = Shape_;
    using WarpMmaSimt = WarpMmaSimt_;
    using OutputOp = OutputOp_;
    static int const kElementsPerAccess = ElementsPerAccess;
    static const int kPartitionsK = Shape::kK / WarpMmaSimt::Shape::kK;

    using ElementOutput = typename OutputOp::ElementOutput;
    using LayoutC = typename WarpMmaSimt::LayoutC;
    using ElementAccumulator = typename WarpMmaSimt::ElementC;

    using OutputTileThreadMap =
        typename cutlass::epilogue::threadblock::DefaultThreadMapSimt<
            Shape, typename WarpMmaSimt::Shape, typename WarpMmaSimt::Policy,
            kPartitionsK, ElementOutput, kElementsPerAccess>::Type;

    using OutputTileIterator =
        cutlass::epilogue::threadblock::PredicatedTileIterator<
            OutputTileThreadMap, ElementOutput>;

    using AccumulatorFragmentIterator =
        cutlass::epilogue::warp::FragmentIteratorSimt<
            typename WarpMmaSimt::Shape, typename WarpMmaSimt::ThreadMma,
            cutlass::layout::RowMajor, typename WarpMmaSimt::Policy>;

    using WarpTileIterator = cutlass::epilogue::warp::TileIteratorSimt<
        typename WarpMmaSimt::Shape, typename WarpMmaSimt::ThreadMma,
        ElementAccumulator, cutlass::layout::RowMajor,
        typename WarpMmaSimt::Policy>;

    using SharedLoadIterator =
        cutlass::epilogue::threadblock::SharedLoadIterator<
            typename OutputTileThreadMap::CompactedThreadMap,
            ElementAccumulator>;

    /// Hard-coded padding elements added
    using Padding = typename WarpTileIterator::Padding;

    using Epilogue =
        EpilogueReduction<Shape, WarpMmaSimt, kPartitionsK, OutputTileIterator,
                          AccumulatorFragmentIterator, WarpTileIterator,
                          SharedLoadIterator, OutputOp, Padding>;
};