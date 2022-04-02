#pragma once
#include "cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"
#include "utils.h"
#include <cutlass/cutlass.h>

template <typename Shape_, typename Layout_, int AdvanceRank,
          typename ThreadMap_, bool Transpose = false>
class QuantizeTileIterator2dThreadTile {
  public:
    static_assert(
        AdvanceRank == 0 || AdvanceRank == 1,
        "Specialization for pitch-linear iterator may along advance along the "
        "contiguous(rank=0) or strided(rank=1) dimension.");

    using Shape = Shape_;
    using Layout = Layout_;
    using Element = int8_t; // output type
    using LoadElement = float;
    static int const kAdvanceRank = AdvanceRank;
    using ThreadMap = ThreadMap_;

    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;

    using TensorRef = cutlass::TensorRef<Element, Layout>;
    using TensorView = cutlass::TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;

    using LoadingIterator =
        cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
            Shape, LoadElement, Layout, AdvanceRank, ThreadMap, Transpose>;
    using StoreIterator =
        cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
            Shape, Element, Layout, AdvanceRank, ThreadMap, Transpose>;

    /// StoreIterator is in the type of computing.
    using AccessType = typename StoreIterator::AccessType;

    /// Fragment object to be loaded or stored
    /// This type is used outside so the type should be quantized
    using Fragment =
        cutlass::Array<Element, ThreadMap::Iterations::kCount *
                                    ThreadMap::ThreadAccessShape::kCount>;
    using LoadFragment = typename LoadingIterator::Fragment;
    using StoreFragment = typename StoreIterator::Fragment;
    static_assert(std::is_same<Fragment, StoreFragment>::value,
                  "Fragment type should be the same");

    /// Parameters object is precomputed state and is host-constructible
    class Params {
      private:
        friend QuantizeTileIterator2dThreadTile;

        /// Parameters object
        typename LoadingIterator::Params load_params_;
        typename StoreIterator::Params store_params_;

        /// scale pointer
        LoadElement *scale_ptr_;

      public:
        CUTLASS_HOST_DEVICE
        Params() {}

        /// Construct the Params object given a pitch-linear tensor's layout
        CUTLASS_HOST_DEVICE
        Params(Layout const &layout, void *scale_ptr)
            : load_params_(layout), store_params_(layout),
              scale_ptr_((LoadElement *)scale_ptr) {}
    };

  private:
    //
    // Data members
    //

    /// Underlying tile iterator
    LoadingIterator load_iterator_;
    StoreIterator store_iterator_;
    /// Pointer to scale
    LoadElement div_scale_;

  public:
    /// Constructs a TileIterator from its precomputed state, threadblock
    /// offset, and thread ID
    CUTLASS_HOST_DEVICE
    QuantizeTileIterator2dThreadTile(
        Params const &params, ///< Precomputed parameters object
        LoadElement *load_pointer, Element *store_pointer,
        TensorCoord extent, ///< Extent of tensor
        int thread_id,      ///< ID of each participating thread
        TensorCoord const &threadblock_offset ///< Initial offset of threadblock
        )
        : load_iterator_(params.load_params_, load_pointer, extent, thread_id,
                         threadblock_offset),
          store_iterator_(params.store_params_, store_pointer, extent,
                          thread_id, threadblock_offset) {
        div_scale_ = 1.0f / __ldg(params.scale_ptr_);
    }

    /// Construct a QuantizeTileIterator2dThreadTile with zero threadblock
    /// offset
    CUTLASS_HOST_DEVICE
    QuantizeTileIterator2dThreadTile(
        Params const &params, ///< Precomputed parameters object
        LoadElement *load_pointer, Element *store_pointer,
        TensorCoord extent, ///< Extent of tensor
        int thread_id       ///< ID of each participating thread
        )
        : QuantizeTileIterator2dThreadTile(params, load_pointer, store_pointer,
                                           extent, thread_id,
                                           cutlass::make_Coord(0, 0)) {}

    /// Adds a pointer offset in units of Element
    CUTLASS_HOST_DEVICE
    void add_pointer_offset(LongIndex pointer_offset) {
        load_iterator_.add_pointer_offset(pointer_offset);
        store_iterator_.add_pointer_offset(pointer_offset);
    }

    /// Advances to the next tile in memory.
    ///
    /// The first time this method is called, predicates are updated, and
    /// the iterator's internal pointer is reverted to the first "steady
    /// state" tile. Subsequent calls are lightweight and must only update
    /// the internal pointer.
    CUTLASS_HOST_DEVICE
    QuantizeTileIterator2dThreadTile &operator++() {
        ++load_iterator_;
        ++store_iterator_;
        return *this;
    }

    /// Advances to the next tile in memory.
    ///
    /// The first time this method is called, predicates are updated, and
    /// the iterator's internal pointer is reverted to the first "steady
    /// state" tile. Subsequent calls are lightweight and must only update
    /// the internal pointer.
    CUTLASS_HOST_DEVICE
    QuantizeTileIterator2dThreadTile operator++(int) {
        QuantizeTileIterator2dThreadTile self(*this);
        operator++();
        return self;
    }

    /// Clears the predicate set efficiently
    CUTLASS_HOST_DEVICE
    void clear_mask(bool enable = true) {
        load_iterator_.clear_mask(enable);
        store_iterator_.clear_mask(enable);
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
        LoadFragment load_frag;
        load_iterator_.load_with_pointer_offset(load_frag, pointer_offset);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < load_frag.size(); ++i) {
            frag[i] = (int8_t)(load_frag[i] * div_scale_);
        }
        store_iterator_.store_with_pointer_offset(frag, pointer_offset);
    }

    /// Loads a fragment from memory
    CUTLASS_DEVICE
    void load(Fragment &frag) { load_with_pointer_offset(frag, 0); }
};
