/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_osp_algorithms.cpp
 * \brief Unit test for OSP Algorithm files.
 */

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <vector>
#include <string>
#include "gtest/gtest.h"

#define MODULE_NAME "OspTests"

#include "passes/algorithms/osp/auxiliary/datastructures/union_find_universe.h"
#include "passes/algorithms/osp/auxiliary/datastructures/heaps/pairing_heap.h"
#include "passes/algorithms/osp/auxiliary/balanced_coin_flips.h"
#include "passes/algorithms/osp/bsp/model/bsp_architecture.h"
#include "passes/algorithms/osp/bsp/model/bsp_instance.h"
#include "passes/algorithms/osp/bsp/model/bsp_schedule.h"
#include "passes/algorithms/osp/bsp/model/util/set_schedule.h"
#include "passes/algorithms/osp/bsp/scheduler/greedy_schedulers/greedy_children.h"
#include "passes/algorithms/osp/bsp/scheduler/greedy_schedulers/greedy_meta_scheduler.h"
#include "passes/algorithms/osp/bsp/scheduler/greedy_schedulers/grow_local_auto_cores.h"
#include "passes/algorithms/osp/bsp/scheduler/improvement_scheduler.h"
#include "passes/algorithms/osp/bsp/scheduler/local_search/kernighan_lin/kl_improver.h"
#include "passes/algorithms/osp/bsp/scheduler/local_search/kernighan_lin/comm_cost_modules/kl_hyper_total_comm_cost.h"
#include "passes/algorithms/osp/coarser/multilevel_coarser.h"
#include "passes/algorithms/osp/dag_divider/isomorphism_divider/isomorphic_subgraph_scheduler.h"
#include "passes/algorithms/osp/dag_divider/isomorphism_divider/merkle_hash_computer.h"
#include "passes/algorithms/osp/dag_divider/isomorphism_divider/precomputed_hash_computer.h"
#include "passes/algorithms/osp/dag_divider/isomorphism_divider/trimmed_group_scheduler.h"
#include "passes/algorithms/osp/coarser/coarser_util.h"
#include "passes/algorithms/osp/coarser/sarkar/sarkar.h"
#include "passes/algorithms/osp/coarser/sarkar/sarkar_mul.h"
#include "passes/algorithms/osp/concepts/directed_graph_edge_desc_concept.h"
#include "passes/algorithms/osp/graph_algorithms/directed_graph_path_util.h"
#include "passes/algorithms/osp/graph_implementations/adj_list_impl/compact_sparse_graph.h"
#include "passes/algorithms/osp/graph_implementations/adj_list_impl/dag_vector_adapter.h"
#include "passes/algorithms/osp/graph_implementations/adj_list_impl/cdag_vertex_impl.h"
#include "passes/algorithms/osp/graph_implementations/adj_list_impl/computational_dag_vector_impl.h"
#include "passes/algorithms/osp/graph_implementations/integral_range.h"

namespace npu::tile_fwk {
namespace osp {

using VertType = int32_t;
using WorkType = int32_t;
using VTypeType = unsigned;

using GraphType = CompactSparseGraph<VertType, VertType, WorkType, WorkType, WorkType, VTypeType>;

using VertexImpl = osp::CDagVertexImpl<VertType, WorkType, WorkType, WorkType, VTypeType>;
using GraphAdapterType = osp::DagVectorAdapter<VertexImpl>;
using ConstrGraphType = osp::ComputationalDagVectorImpl<VertexImpl>;

class TestOspAlgorithms : public testing::Test {
public:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override {}
};

class TestMultilevelCoarser : public MultilevelCoarser<GraphType, GraphType> {
public:
    enum class Scenario { NO_CONTRACTIONS, COMPACT_IDENTITY, KEEP_LARGE_REDUCTION, PREBUILT_GRAPH };

    explicit TestMultilevelCoarser(Scenario scenario) : scenario_(scenario) {}

    [[nodiscard]] std::size_t HistorySize() const { return dagHistory_.size(); }

protected:
    ReturnStatus RunContractions() override
    {
        if (scenario_ == Scenario::NO_CONTRACTIONS) {
            return ReturnStatus::OSP_SUCCESS;
        }

        auto identityMap = [this]() {
            const std::size_t size = dagHistory_.empty() ? GetOriginalGraph()->NumVertices() :
                                                           dagHistory_.back()->NumVertices();
            std::vector<VertexIdxT<GraphType>> contractionMap(size);
            std::iota(contractionMap.begin(), contractionMap.end(), 0);
            return contractionMap;
        };

        if (scenario_ == Scenario::PREBUILT_GRAPH) {
            auto contractionMap = identityMap();
            GraphType contractedGraph;
            if (!coarser_util::ConstructCoarseDag(*GetOriginalGraph(), contractedGraph, contractionMap)) {
                return ReturnStatus::OSP_ERROR;
            }
            return AddContraction(std::move(contractionMap), std::move(contractedGraph));
        }

        ReturnStatus status = AddContraction(identityMap());
        if (scenario_ == Scenario::KEEP_LARGE_REDUCTION) {
            auto contractionMap = identityMap();
            for (std::size_t i = 0; i < contractionMap.size(); ++i) {
                contractionMap[i] = static_cast<VertexIdxT<GraphType>>(i / 3U);
            }
            status = std::max(status, AddContraction(std::move(contractionMap)));
        } else {
            status = std::max(status, AddContraction(identityMap()));
        }
        status = std::max(status, AddContraction(identityMap()));
        return status;
    }

private:
    Scenario scenario_;
};

TEST_F(TestOspAlgorithms, UnionFind1)
{
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f"});
    UnionFindUniverse<std::string, unsigned, int> testUniverse;
    for (const auto& name : names) {
        testUniverse.AddObject(name);
    }

    for (auto& name : names) {
        EXPECT_EQ(testUniverse.FindOriginByName(name), name);
    }

    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 6);

    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 6);

    testUniverse.JoinByName("a", "b");
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 5);

    testUniverse.JoinByName("b", "c");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 4);
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("c"));
    EXPECT_EQ(testUniverse.FindOriginByName("b"), testUniverse.FindOriginByName("c"));

    testUniverse.JoinByName("d", "b");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 3);
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("d"));
    EXPECT_EQ(testUniverse.FindOriginByName("d"), testUniverse.FindOriginByName("b"));

    testUniverse.JoinByName("a", "c");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 3);
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("c"));

    testUniverse.JoinByName("a", "d");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 3);
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("d"));

    testUniverse.JoinByName("e", "f");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 2);
    EXPECT_EQ(testUniverse.FindOriginByName("e"), testUniverse.FindOriginByName("f"));
    EXPECT_NE(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("f"));

    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    EXPECT_EQ(testUniverse.FindOriginByName("b"), testUniverse.FindOriginByName("c"));
    EXPECT_EQ(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("d"));
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("d"));
    EXPECT_EQ(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("b"));

    EXPECT_EQ(testUniverse.FindOriginByName("e"), testUniverse.FindOriginByName("f"));

    EXPECT_NE(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("f"));
}

TEST_F(TestOspAlgorithms, UnionFind2)
{
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f", "g", "h", "i"});
    UnionFindUniverse<std::string, unsigned, int> testUniverse;

    for (auto& name : names) {
        testUniverse.AddObject(name);
    }

    for (auto& name : names) {
        EXPECT_EQ(testUniverse.FindOriginByName(name), name);
    }

    for (auto& name : names) {
        EXPECT_EQ(testUniverse.FindOriginByName(name), name);
    }

    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 9);

    testUniverse.JoinByName("a", "b");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 8);
    testUniverse.JoinByName("b", "c");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 7);
    testUniverse.JoinByName("c", "d");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 6);
    testUniverse.JoinByName("d", "e");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 5);
    testUniverse.JoinByName("e", "f");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 4);

    testUniverse.JoinByName("c", "f");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 4);

    testUniverse.JoinByName("g", "h");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 3);
    testUniverse.JoinByName("h", "i");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 2);

    testUniverse.JoinByName("b", "h");
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 1);

    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    EXPECT_EQ(testUniverse.FindOriginByName("b"), testUniverse.FindOriginByName("c"));
    EXPECT_EQ(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("d"));
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("h"));
    EXPECT_EQ(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("i"));
    EXPECT_EQ(testUniverse.FindOriginByName("f"), testUniverse.FindOriginByName("g"));

    testUniverse.Reset();
    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 0);
}

TEST_F(TestOspAlgorithms, UnionFind3)
{
    std::vector<std::string> names({"a", "b", "c", "d", "e", "f"});
    std::vector<unsigned> weights({1, 2, 1, 3, 1, 1});

    UnionFindUniverse<std::string, unsigned, unsigned> testUniverse;
    for (std::size_t i = 0; i < names.size(); ++i) {
        testUniverse.AddObject(names[i], weights[i]);
    }

    for (size_t i = 0; i < names.size(); i++) {
        EXPECT_EQ(testUniverse.FindOriginByName(names[i]), names[i]);
        EXPECT_EQ(testUniverse.GetWeightOfComponentByName(names[i]), weights[i]);
    }

    EXPECT_EQ(testUniverse.GetNumberOfConnectedComponents(), 6);

    testUniverse.JoinByName("a", "b");
    testUniverse.JoinByName("b", "c");
    testUniverse.JoinByName("d", "b");
    testUniverse.JoinByName("a", "c");
    testUniverse.JoinByName("a", "d");

    testUniverse.JoinByName("e", "f");

    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("b"));
    EXPECT_EQ(testUniverse.FindOriginByName("b"), testUniverse.FindOriginByName("c"));
    EXPECT_EQ(testUniverse.FindOriginByName("c"), testUniverse.FindOriginByName("d"));
    EXPECT_EQ(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("d"));

    EXPECT_EQ(testUniverse.FindOriginByName("e"), testUniverse.FindOriginByName("f"));

    EXPECT_NE(testUniverse.FindOriginByName("a"), testUniverse.FindOriginByName("f"));

    EXPECT_EQ(testUniverse.GetWeightOfComponentByName("a"), 7);
    EXPECT_EQ(testUniverse.GetWeightOfComponentByName("b"), 7);

    std::vector<std::vector<std::string>> components = testUniverse.GetConnectedComponents();
    unsigned totalCompWeights = 0;
    unsigned totalElements = 0;
    for (auto& comp : components) {
        totalCompWeights += testUniverse.GetWeightOfComponentByName(comp.at(0));
        totalElements += static_cast<unsigned>(comp.size());
        for (auto& name : comp) {
            EXPECT_TRUE(std::any_of(names.cbegin(), names.cend(),
                                    [name](std::string other_name) { return name == other_name; }));
        }
    }

    unsigned totalWeight = 0;
    for (auto& wt : weights) {
        totalWeight += wt;
    }

    EXPECT_EQ(totalElements, names.size());
    EXPECT_EQ(totalWeight, totalCompWeights);

    for (auto& name : names) {
        EXPECT_TRUE(std::any_of(components.cbegin(), components.cend(), [name](std::vector<std::string> comp) {
            return std::any_of(comp.cbegin(), comp.cend(),
                               [name](std::string other_name) { return name == other_name; });
        }));
    }
}

// Verify public heap operations, including updates, equal maxima, and invalid keys.
TEST_F(TestOspAlgorithms, PairingHeapOperations)
{
    MaxPairingHeap<int, int> heap;
    EXPECT_TRUE(heap.IsEmpty());
    EXPECT_TRUE(heap.GetTopKeys().empty());
    EXPECT_THROW(heap.Pop(), std::runtime_error);

    heap.Push(1, 10);
    heap.Push(2, 20);
    heap.Push(3, 15);
    heap.Push(4, 5);
    EXPECT_EQ(heap.size(), 4U);
    EXPECT_EQ(heap.Top(), 2);
    EXPECT_EQ(heap.GetValue(4), 5);
    EXPECT_THROW(heap.Push(2, 30), std::invalid_argument);

    heap.Update(1, 25);
    EXPECT_EQ(heap.Top(), 1);
    heap.Update(2, 0);
    EXPECT_EQ(heap.GetValue(2), 0);

    heap.Push(5, 25);
    auto topKeys = heap.GetTopKeys();
    std::sort(topKeys.begin(), topKeys.end());
    EXPECT_EQ(topKeys, std::vector<int>({1, 5}));
    EXPECT_EQ(heap.GetTopKeys(1).size(), 1U);

    heap.Erase(3);
    EXPECT_FALSE(heap.Contains(3));
    EXPECT_THROW(heap.Update(3, 30), std::invalid_argument);
    EXPECT_THROW(heap.Erase(3), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(heap.GetValue(3)), std::invalid_argument);

    heap.Clear();
    EXPECT_TRUE(heap.IsEmpty());
    EXPECT_EQ(heap.size(), 0U);
}

TEST_F(TestOspAlgorithms, IntSqrt)
{
    EXPECT_EQ(IntSqrtFloor(0), 0);
    EXPECT_EQ(IntSqrtFloor(-1), 0);

    for (std::size_t root = 1U; root < 200U; ++root) {
        for (std::size_t num = root * root; num < (root + 1U) * (root + 1U); ++num) {
            EXPECT_EQ(IntSqrtFloor(num), root);
        }
    }

    for (int root = 1; root < 300; ++root) {
        for (int num = root * root; num < (root + 1) * (root + 1); ++num) {
            EXPECT_EQ(IntSqrtFloor(num), root);
        }
    }
}

TEST_F(TestOspAlgorithms, Divisors)
{
    EXPECT_EQ(DivisorsList(0), std::vector<int>({0}));
    EXPECT_TRUE(DivisorsList(-1).empty());

    for (std::size_t num = 1U; num < 1000U; ++num) {
        const std::vector<std::size_t> divs = DivisorsList(num);
        for (const std::size_t& div : divs) {
            EXPECT_EQ(num % div, 0U);
        }

        auto it = divs.begin();
        for (std::size_t i = 1U; i <= num; ++i) {
            if (num % i == 0) {
                EXPECT_TRUE(it != divs.end());
                EXPECT_EQ(i, *it);
                ++it;
            }
        }
        EXPECT_TRUE(it == divs.end());
    }
}

bool thueMorseGen(long unsigned int n)
{
    unsigned long int bin_sum = 0;
    while (n != 0) {
        bin_sum += n % 2;
        n /= 2;
    }
    return bool(bin_sum % 2);
}

TEST_F(TestOspAlgorithms, RandomBiasedCoin)
{
    BiasedRandom Coin;
    bool valAnd = true;
    bool valOr = false;
    for (int i = 0; i < 1000; i++) {
        const bool flip = Coin.GetFlip();
        valAnd &= flip;
        valOr |= flip;
    }

    // Can both technically fail but insanely unlikely
    EXPECT_FALSE(valAnd);
    EXPECT_TRUE(valOr);
}

TEST_F(TestOspAlgorithms, ThueMorse)
{
    ThueMorseSequence coin(0);

    std::vector<bool> beginning(
        {0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1});
    std::vector<bool> generated;
    for (long unsigned i = 0; i < beginning.size(); i++) {
        const bool next = coin.GetFlip();
        generated.emplace_back(next);
    }

    EXPECT_TRUE(beginning == generated);

    ThueMorseSequence testCoinInSeq(0);
    for (unsigned i = 0; i < 200u; i++) {
        EXPECT_EQ(testCoinInSeq.GetFlip(), thueMorseGen(i));
    }
}

TEST_F(TestOspAlgorithms, InPlaceInversePermutationRandom)
{
    std::vector<unsigned> vec(20);
    std::iota(vec.begin(), vec.end(), 0);
    std::vector<unsigned> sol(vec);

    std::random_device rd;
    std::mt19937 gen(rd());

    for (unsigned i = 0; i < 5U; ++i) {
        std::shuffle(vec.begin(), vec.end(), gen);

        std::vector<unsigned> invPerm(vec.size());
        for (unsigned j = 0; j < vec.size(); ++j) {
            invPerm[vec[j]] = j;
        }

        InversePermuteInplace(vec, invPerm);
        for (std::size_t j = 0; j < sol.size(); ++j) {
            EXPECT_EQ(vec[j], sol[j]);
        }
    }
}

TEST_F(TestOspAlgorithms, InPlaceInversePermutationChar)
{
    std::vector<char> vec({'a', 'b', 'c', 'd', 'e', 'f', 'g'});
    std::vector<std::size_t> perm({4, 0, 1, 2, 3, 6, 5});
    std::vector<char> sol({'e', 'a', 'b', 'c', 'd', 'g', 'f'});

    InversePermuteInplace(vec, perm);
    for (std::size_t j = 0; j < sol.size(); ++j) {
        EXPECT_EQ(vec[j], sol[j]);
    }
}

TEST_F(TestOspAlgorithms, Architecture)
{
    std::vector<std::vector<WorkType>> uniformSentCosts = {{0, 1, 1, 1}, {1, 0, 1, 1}, {1, 1, 0, 1}, {1, 1, 1, 0}};

    BspArchitecture<GraphType> architecture;
    architecture.SetNumberOfProcessors(4);
    architecture.SetCommunicationCosts(2);
    architecture.SetSynchronisationCosts(3);

    EXPECT_EQ(architecture.NumberOfProcessors(), 4);
    EXPECT_EQ(architecture.CommunicationCosts(), 2);
    EXPECT_EQ(architecture.SynchronisationCosts(), 3);
    EXPECT_EQ(architecture.GetMemoryConstraintType(), MemoryConstraintType::NONE);
    EXPECT_EQ(architecture.GetNumberOfProcessorTypes(), 1);

    EXPECT_EQ(architecture.MemoryBound(0), 100);
    EXPECT_EQ(architecture.MemoryBound(1), 100);
    EXPECT_EQ(architecture.MemoryBound(2), 100);
    EXPECT_EQ(architecture.MemoryBound(3), 100);
    architecture.SetMemoryBound(200, 3);
    EXPECT_EQ(architecture.MemoryBound(2), 100);
    EXPECT_EQ(architecture.MemoryBound(3), 200);

    EXPECT_EQ(architecture.ProcessorTypes()[0], 0);
    EXPECT_EQ(architecture.ProcessorTypes()[1], 0);
    EXPECT_EQ(architecture.ProcessorTypes()[2], 0);
    EXPECT_EQ(architecture.ProcessorTypes()[3], 0);

    EXPECT_EQ(architecture.ProcessorType(0), 0);
    EXPECT_EQ(architecture.ProcessorType(1), 0);
    EXPECT_EQ(architecture.ProcessorType(2), 0);
    EXPECT_EQ(architecture.ProcessorType(3), 0);
    architecture.SetProcessorsWithTypes({0, 0, 0, 1});
    EXPECT_EQ(architecture.ProcessorType(2), 0);
    EXPECT_EQ(architecture.ProcessorType(3), 1);

    EXPECT_EQ(architecture.CommunicationCosts(0, 1), 2);
    EXPECT_EQ(architecture.CommunicationCosts(0, 0), 0);

    EXPECT_EQ(architecture.GetNumberOfProcessorTypes(), 2);

    EXPECT_TRUE(architecture.SendCost() == uniformSentCosts);
    EXPECT_EQ(architecture.CommunicationCosts(0, 1), 2);
    EXPECT_EQ(architecture.CommunicationCosts(0, 0), 0);
}

// Reject invalid processor, memory, and send-cost configurations at the public API boundary.
TEST_F(TestOspAlgorithms, ArchitectureRejectsInvalidConfigurations)
{
    using SendCosts = std::vector<std::vector<WorkType>>;

    EXPECT_THROW(BspArchitecture<GraphType>(0U), std::runtime_error);
    EXPECT_THROW(BspArchitecture<GraphType>(2U, 1, 2, 100, SendCosts({{0, 1}})), std::invalid_argument);
    EXPECT_THROW(BspArchitecture<GraphType>(2U, 1, 2, 100, SendCosts({{0, 1}, {1}})), std::invalid_argument);

    BspArchitecture<GraphType> architecture(2U, 1, 2, 100, SendCosts({{9, 3}, {4, 8}}));
    EXPECT_EQ(architecture.SendCosts(0U, 0U), 0);
    EXPECT_EQ(architecture.SendCosts(0U, 1U), 3);
    EXPECT_THROW(architecture.SetMemoryBound(std::vector<WorkType>({100})), std::invalid_argument);
    EXPECT_THROW(architecture.SetNumberOfProcessors(0U), std::invalid_argument);
    EXPECT_THROW(architecture.SetProcessorsWithTypes({}), std::invalid_argument);
    EXPECT_THROW(architecture.SetProcessorsConsequTypes({1U, 1U}, {100}), std::invalid_argument);
}

// Exercise malformed assignments and each schedule validity constraint independently.
TEST_F(TestOspAlgorithms, ScheduleRejectsInvalidAssignments)
{
    const std::set<std::pair<VertType, VertType>> edges({{0, 1}});
    BspInstance<GraphType> instance;
    instance.GetArchitecture() = BspArchitecture<GraphType>(2U);
    instance.GetComputationalDag() = GraphType(2, edges);

    BspSchedule<GraphType> malformed(instance, {0U}, {0U, 0U});
    EXPECT_FALSE(malformed.SatisfiesPrecedenceConstraints());
    EXPECT_FALSE(malformed.SatisfiesNodeTypeConstraints());

    BspSchedule<GraphType> schedule(instance, {0U, 0U}, {0U, 0U});
    EXPECT_THROW(schedule.SetAssignedSuperstep(2, 0U), std::invalid_argument);
    schedule.SetAssignedSuperstep(1, 2U);
    schedule.SetNumberOfSupersteps(1U);
    EXPECT_FALSE(schedule.SatisfiesPrecedenceConstraints());

    schedule = BspSchedule<GraphType>(instance, {0U, 2U}, {0U, 1U});
    EXPECT_FALSE(schedule.SatisfiesPrecedenceConstraints());

    schedule = BspSchedule<GraphType>(instance, {0U, 1U}, {0U, 0U});
    EXPECT_FALSE(schedule.SatisfiesPrecedenceConstraints());

    instance.GetArchitecture().SetProcessorsWithTypes({0U, 1U});
    instance.GetComputationalDag().SetVertexType(1, 1U);
    instance.SetDiagonalCompatibilityMatrix(2U);
    schedule = BspSchedule<GraphType>(instance, {0U, 0U}, {0U, 1U});
    EXPECT_FALSE(schedule.SatisfiesNodeTypeConstraints());
}

TEST_F(TestOspAlgorithms, EmptyGraph)
{
    GraphType graph;

    EXPECT_EQ(graph.NumVertices(), 0);
    EXPECT_EQ(graph.NumEdges(), 0);
}

TEST_F(TestOspAlgorithms, NoEdgesGraph)
{
    const std::vector<std::pair<VertType, VertType>> edges({});

    GraphType graph(10, edges);

    EXPECT_EQ(graph.NumVertices(), 10);
    EXPECT_EQ(graph.NumEdges(), 0);
}

GraphType LineGraph()
{
    const std::set<std::pair<VertType, VertType>> edges({{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}});

    return GraphType(8, edges);
}

TEST_F(TestOspAlgorithms, TestLineGraph)
{
    const GraphType graph = LineGraph();

    EXPECT_EQ(graph.NumVertices(), 8);
    EXPECT_EQ(graph.NumEdges(), 7);

    std::size_t cntr = 0;
    for (const auto& vert : graph.Vertices()) {
        EXPECT_EQ(vert, cntr);
        ++cntr;
    }
    EXPECT_EQ(graph.NumVertices(), cntr);

    for (const auto& vert : graph.Vertices()) {
        VertType notLast = static_cast<VertType>((vert != 7));

        EXPECT_EQ(graph.OutDegree(vert), notLast);
        for (const auto& chld : graph.Children(vert)) {
            EXPECT_EQ(chld, vert + notLast);
        }
        auto chldren = graph.Children(vert);
        EXPECT_EQ(chldren.crend() - chldren.crbegin(), graph.OutDegree(vert));
        for (auto it = chldren.crbegin(); it != chldren.crend(); ++it) {
            EXPECT_EQ(*it, vert + notLast);
        }
    }
    for (const auto& vert : graph.Vertices()) {
        VertType notFirst = static_cast<VertType>((vert != 0));

        EXPECT_EQ(graph.InDegree(vert), notFirst);
        for (const auto& par : graph.Parents(vert)) {
            EXPECT_EQ(par, vert - notFirst);
        }
        auto prnts = graph.Parents(vert);
        EXPECT_EQ(prnts.crend() - prnts.crbegin(), graph.InDegree(vert));
        for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
            EXPECT_EQ(*it, vert - notFirst);
        }
    }

    for (const auto& vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexType(vert), 0);
    }
}

GraphType SimpleGraph()
{
    const std::vector<std::pair<VertType, VertType>> edges(
        {{0, 1}, {2, 3}, {6, 10}, {7, 9}, {0, 2}, {4, 6}, {1, 6}, {6, 7}, {5, 6}, {3, 7}, {1, 2}});

    return GraphType(11, edges);
}

TEST_F(TestOspAlgorithms, Graph1)
{
    const GraphType graph = SimpleGraph();

    EXPECT_EQ(graph.NumVertices(), 11);
    EXPECT_EQ(graph.NumEdges(), 11);

    std::size_t cntr0{};
    std::size_t cntrChldEdges{};
    std::size_t cntrParEdges{};
    for (const auto& vert : graph.Vertices()) {
        EXPECT_EQ(vert, cntr0++);
        cntrChldEdges += graph.OutDegree(vert);
        cntrParEdges += graph.InDegree(vert);
    }
    EXPECT_EQ(graph.NumVertices(), cntr0);
    EXPECT_EQ(graph.NumEdges(), cntrChldEdges);
    EXPECT_EQ(graph.NumEdges(), cntrParEdges);

    const std::vector<std::vector<std::size_t>> outEdges(
        {{1, 2}, {2, 6}, {3}, {7}, {6}, {6}, {7, 10}, {9}, {}, {}, {}});

    for (const auto& vert : graph.Vertices()) {
        EXPECT_EQ(graph.OutDegree(vert), outEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto& chld : graph.Children(vert)) {
            EXPECT_EQ(chld, outEdges[vert][cntr++]);
        }
        auto chldrn = graph.Children(vert);
        EXPECT_EQ(chldrn.crend() - chldrn.crbegin(), graph.OutDegree(vert));
        for (auto it = chldrn.crbegin(); it != chldrn.crend(); ++it) {
            EXPECT_EQ(*it, outEdges[vert][--cntr]);
        }
    }

    const std::vector<std::vector<std::size_t>> inEdges(
        {{}, {0}, {0, 1}, {2}, {}, {}, {1, 4, 5}, {3, 6}, {}, {7}, {6}});

    for (const auto& vert : graph.Vertices()) {
        EXPECT_EQ(graph.InDegree(vert), inEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto& par : graph.Parents(vert)) {
            EXPECT_EQ(par, inEdges[vert][cntr++]);
        }
        auto prnts = graph.Parents(vert);
        EXPECT_EQ(prnts.crend() - prnts.crbegin(), graph.InDegree(vert));
        for (auto it = prnts.crbegin(); it != prnts.crend(); ++it) {
            EXPECT_EQ(*it, inEdges[vert][--cntr]);
        }
    }

    for (const auto& vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexType(vert), 0);
    }
}

TEST_F(TestOspAlgorithms, GraphWorkWeights)
{
    std::vector<unsigned> ww(11);
    std::iota(ww.begin(), ww.end(), 0);

    GraphType graph = SimpleGraph();
    for (auto vert : graph.Vertices()) {
        graph.SetVertexWorkWeight(vert, ww[vert]);
    }

    for (auto vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexWorkWeight(vert), ww[vert]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexWorkWeight(vert, wt);
        EXPECT_EQ(graph.VertexWorkWeight(vert), wt);
    }
}

TEST_F(TestOspAlgorithms, GraphCommWeights)
{
    std::vector<unsigned> cw(11);
    std::iota(cw.begin(), cw.end(), 11);

    GraphType graph = SimpleGraph();
    for (auto vert : graph.Vertices()) {
        graph.SetVertexCommWeight(vert, cw[vert]);
    }

    for (auto vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexCommWeight(vert), cw[vert]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexCommWeight(vert, wt);
        EXPECT_EQ(graph.VertexCommWeight(vert), wt);
    }
}

TEST_F(TestOspAlgorithms, GraphMemWeights)
{
    std::vector<unsigned> mw(11);
    std::iota(mw.begin(), mw.end(), 22);

    GraphType graph = SimpleGraph();

    for (auto vert : graph.Vertices()) {
        graph.SetVertexMemWeight(vert, mw[vert]);
    }

    for (auto vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexMemWeight(vert), mw[vert]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexMemWeight(vert, wt);
        EXPECT_EQ(graph.VertexMemWeight(vert), wt);
    }
}

TEST_F(TestOspAlgorithms, GraphVtype)
{
    std::vector<unsigned> vt(11);
    std::iota(vt.begin(), vt.end(), 33);

    GraphType graph = SimpleGraph();

    for (auto vert : graph.Vertices()) {
        graph.SetVertexType(vert, vt[vert]);
    }

    for (auto vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexType(vert), vt[vert]);

        const unsigned wt = static_cast<unsigned>(rand());
        graph.SetVertexType(vert, wt);
        EXPECT_EQ(graph.VertexType(vert), wt);
    }
}

TEST_F(TestOspAlgorithms, ExpansionMapValidity)
{
    const std::vector<std::vector<VertexIdxT<GraphType>>> expansionmap1 = {{0}, {1}, {2}, {3}};
    EXPECT_TRUE(coarser_util::CheckValidExpansionMap<GraphType>(expansionmap1));

    const std::vector<std::vector<VertexIdxT<GraphType>>> expansionmap2 = {{0}, {2}, {3}};
    EXPECT_FALSE(coarser_util::CheckValidExpansionMap<GraphType>(expansionmap2));

    const std::vector<std::vector<VertexIdxT<GraphType>>> expansionmap3 = {{0, 3}};
    EXPECT_FALSE(coarser_util::CheckValidExpansionMap<GraphType>(expansionmap3));

    const std::vector<std::vector<VertexIdxT<GraphType>>> expansionmap4 = {{0, 3}, {2, 1, 4}, {5}};
    EXPECT_TRUE(coarser_util::CheckValidExpansionMap<GraphType>(expansionmap4));

    const std::vector<std::vector<VertexIdxT<GraphType>>> expansionmap5 = {{0}, {}, {2}, {3}, {1}};
    EXPECT_FALSE(coarser_util::CheckValidExpansionMap<GraphType>(expansionmap5));

    const std::vector<std::vector<VertexIdxT<GraphType>>> expansionmap6 = {{-1}};
    EXPECT_FALSE(coarser_util::CheckValidExpansionMap<GraphType>(expansionmap6));

    const std::vector<std::vector<VertexIdxT<GraphType>>> expansionmap7 = {{0}, {0}};
    EXPECT_FALSE(coarser_util::CheckValidExpansionMap<GraphType>(expansionmap7));
}

TEST_F(TestOspAlgorithms, ContractionMapValidity)
{
    const std::vector<VertexIdxT<GraphType>> contractionMap1 = {0, 1, 2, 3};
    EXPECT_TRUE(coarser_util::CheckValidContractionMap<GraphType>(contractionMap1));

    const std::vector<VertexIdxT<GraphType>> contractionMap2 = {0, 1, 1, 1};
    EXPECT_TRUE(coarser_util::CheckValidContractionMap<GraphType>(contractionMap2));

    const std::vector<VertexIdxT<GraphType>> contractionMap3 = {0, 1, 1, 3};
    EXPECT_FALSE(coarser_util::CheckValidContractionMap<GraphType>(contractionMap3));

    const std::vector<VertexIdxT<GraphType>> contractionMap4 = {2, 1, 1, 3};
    EXPECT_FALSE(coarser_util::CheckValidContractionMap<GraphType>(contractionMap4));
}

TEST_F(TestOspAlgorithms, ContractionMapCoarsening)
{
    std::set<std::pair<VertexIdxT<GraphType>, VertexIdxT<GraphType>>> edges({{0, 1}, {1, 2}});
    GraphType graph(6, edges);

    GraphType coarseGraph1;

    std::vector<VertexIdxT<GraphType>> contractionMap({0, 0, 1, 1, 2, 3});
    EXPECT_TRUE(coarser_util::CheckValidContractionMap<GraphType>(contractionMap));
    EXPECT_TRUE(coarser_util::ConstructCoarseDag(graph, coarseGraph1, contractionMap));
    EXPECT_TRUE(contractionMap == std::vector<VertexIdxT<GraphType>>({0, 0, 1, 1, 2, 3}));

    EXPECT_EQ(coarseGraph1.NumVertices(), 4);
    EXPECT_EQ(coarseGraph1.NumEdges(), 1);

    EXPECT_EQ(coarseGraph1.OutDegree(0), 1);
    EXPECT_EQ(coarseGraph1.OutDegree(1), 0);
    EXPECT_EQ(coarseGraph1.OutDegree(2), 0);

    EXPECT_EQ(coarseGraph1.InDegree(0), 0);
    EXPECT_EQ(coarseGraph1.InDegree(1), 1);
    EXPECT_EQ(coarseGraph1.InDegree(2), 0);

    for (const auto& vert : coarseGraph1.Children(0)) {
        EXPECT_EQ(vert, 1);
    }

    for (const auto& vert : coarseGraph1.Parents(1)) {
        EXPECT_EQ(vert, 0);
    }

    ConstrGraphType emptyCoarseGraph;
    emptyCoarseGraph.AddVertex(1, 1, 1);
    EXPECT_TRUE(coarser_util::ConstructCoarseDag(GraphType(), emptyCoarseGraph, {}));
    EXPECT_TRUE(emptyCoarseGraph.empty());
}

TEST_F(TestOspAlgorithms, TestTopSort)
{
    const GraphType graph = SimpleGraph();

    std::vector<VertexIdxT<GraphType>> verts(graph.NumVertices());
    std::iota(verts.begin(), verts.end(), 0);
    const auto topOrderVec = GetTopOrder<GraphType>(graph);
    EXPECT_TRUE(std::is_permutation(topOrderVec.cbegin(), topOrderVec.cend(), verts.cbegin(), verts.cend()));
    for (const auto vert : graph.Vertices()) {
        for (const auto chld : graph.Children(vert)) {
            EXPECT_GT(std::distance(std::find(topOrderVec.cbegin(), topOrderVec.cend(), vert),
                                    std::find(topOrderVec.cbegin(), topOrderVec.cend(), chld)),
                      0);
        }
    }
}

TEST_F(TestOspAlgorithms, EdgeViewsSupportIndexedIteration)
{
    const GraphType graph(5, std::set<std::pair<VertType, VertType>>({{1, 2}, {1, 4}, {3, 4}}));
    const auto edgeView = Edges(graph);

    ASSERT_EQ(edgeView.size(), 3U);
    auto indexedEdge = EdgeView<GraphType>::Iterator(1, graph);
    EXPECT_EQ(Source(*indexedEdge, graph), 1);
    EXPECT_EQ(Target(*indexedEdge, graph), 4);

    const auto previousEdge = indexedEdge++;
    EXPECT_EQ(previousEdge->source_, 1);
    EXPECT_EQ(indexedEdge->source_, 3);
    EXPECT_EQ(indexedEdge->target_, 4);
    EXPECT_EQ(++indexedEdge, edgeView.end());

    const GraphType emptyGraph(2, std::set<std::pair<VertType, VertType>>({}));
    EXPECT_TRUE(Edges(emptyGraph).empty());
    EXPECT_EQ(Edges(emptyGraph).begin(), Edges(emptyGraph).end());
}

TEST_F(TestOspAlgorithms, TopSortRejectsCyclesAndSupportsReverseOrder)
{
    ConstrGraphType cycle;
    for (unsigned i = 0; i < 3U; ++i) {
        cycle.AddVertex(1, 1, 1);
    }
    ASSERT_TRUE(cycle.AddEdge(0, 1));
    ASSERT_TRUE(cycle.AddEdge(1, 2));
    ASSERT_TRUE(cycle.AddEdge(2, 0));
    EXPECT_THROW(static_cast<void>(GetTopOrder(cycle)), std::runtime_error);

    const auto reverseOrder = GetTopOrderReverse(SimpleGraph());
    ASSERT_FALSE(reverseOrder.empty());
    EXPECT_EQ(reverseOrder.front(), 10);
    EXPECT_EQ(reverseOrder.back(), 0);
}

TEST_F(TestOspAlgorithms, GraphUtilitiesHandleEmptyAndAbsentEdges)
{
    const GraphType emptyGraph;
    EXPECT_EQ(CriticalPathWeight(emptyGraph), 0);

    const GraphType graph(2, std::set<std::pair<VertType, VertType>>({{0, 1}}));
    EXPECT_TRUE(Edge(0, 1, graph));
    EXPECT_FALSE(Edge(1, 0, graph));
    EXPECT_TRUE(EdgeDesc(0, 1, graph).second);
    EXPECT_FALSE(EdgeDesc(1, 0, graph).second);
}

TEST_F(TestOspAlgorithms, NodeDistances)
{
    const std::vector<unsigned> botDistAns = {5, 4, 3, 2, 3, 3, 2, 1, 0, 0, 0};
    const std::vector<unsigned> topDistAns = {0, 1, 2, 3, 0, 0, 2, 4, 0, 5, 3};

    const GraphType graph = SimpleGraph();
    const std::vector<unsigned> botDist = GetBottomNodeDistance(graph);
    const std::vector<unsigned> topDist = GetTopNodeDistance(graph);

    EXPECT_EQ(botDist.size(), graph.NumVertices());
    EXPECT_EQ(topDist.size(), graph.NumVertices());

    for (const auto vert : graph.Vertices()) {
        EXPECT_EQ(botDist[vert], botDistAns[vert]);
        EXPECT_EQ(topDist[vert], topDistAns[vert]);
    }
}

TEST_F(TestOspAlgorithms, TestIntegralRange)
{
    const std::size_t length0 = 0U;
    for (const auto val : IntegralRange(length0)) {
        EXPECT_TRUE(false);
        EXPECT_EQ(val, 100U);
    }

    const std::size_t length147 = 147U;
    std::size_t cntr147 = 0U;
    for (const auto val : IntegralRange(length147)) {
        EXPECT_EQ(val, cntr147++);
    }
    EXPECT_EQ(cntr147, length147);

    const std::size_t start67 = 67U;
    const std::size_t end67 = 67U;
    for (const auto val : IntegralRange(start67, end67)) {
        EXPECT_TRUE(false);
        EXPECT_EQ(val, 100U);
    }

    std::size_t start134 = 67U;
    const std::size_t end257 = 67U;
    for (const auto val : IntegralRange(start134, end257)) {
        EXPECT_EQ(val, start134++);
    }
    EXPECT_EQ(start134, end257);
}

template <typename OtherGraphType>
void testValidContractionMap(const GraphType& graph, const OtherGraphType& coarseGraph,
                             const std::vector<VertexIdxT<GraphType>>& contractionMap)
{
    EXPECT_EQ(contractionMap.size(), graph.NumVertices());
    EXPECT_TRUE(coarser_util::CheckValidContractionMap<GraphType>(contractionMap));

    for (auto vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexType(vert), coarseGraph.VertexType(contractionMap[vert]));
    }

    // Acyclic check
    std::vector<VertexIdxT<GraphType>> coarseVerts(coarseGraph.NumVertices());
    std::iota(coarseVerts.begin(), coarseVerts.end(), 0);
    const auto coarseTopOrder = GetTopOrder<OtherGraphType>(coarseGraph);
    EXPECT_TRUE(
        std::is_permutation(coarseTopOrder.cbegin(), coarseTopOrder.cend(), coarseVerts.cbegin(), coarseVerts.cend()));
    for (const auto vert : coarseGraph.Vertices()) {
        for (const auto chld : coarseGraph.Children(vert)) {
            EXPECT_GT(std::distance(std::find(coarseTopOrder.cbegin(), coarseTopOrder.cend(), vert),
                                    std::find(coarseTopOrder.cbegin(), coarseTopOrder.cend(), chld)),
                      0);
        }
    }
    for (const auto vert : graph.Vertices()) {
        for (const auto chld : graph.Children(vert)) {
            EXPECT_GE(std::distance(std::find(coarseTopOrder.cbegin(), coarseTopOrder.cend(), contractionMap[vert]),
                                    std::find(coarseTopOrder.cbegin(), coarseTopOrder.cend(), contractionMap[chld])),
                      0);
        }
    }

    // Grouping of Vertex Types
    std::vector<unsigned> coarseTypes(coarseGraph.NumVertices(), std::numeric_limits<unsigned>::max());
    for (const auto vert : graph.Vertices()) {
        const auto coarseVert = contractionMap[vert];
        if (coarseTypes[coarseVert] != std::numeric_limits<unsigned>::max()) {
            EXPECT_EQ(coarseTypes[coarseVert], graph.VertexType(vert));
        }
        coarseTypes[coarseVert] = graph.VertexType(vert);
    }
}

GraphType SimpleGraphWithVertexTypes()
{
    std::vector<unsigned> vt(11, 0);
    vt[0] = 1U;
    vt[1] = 1U;
    vt[2] = 1U;

    GraphType graph = SimpleGraph();
    for (auto vert : graph.Vertices()) {
        graph.SetVertexType(vert, vt[vert]);
    }

    return graph;
}

void testCoarseningAlgorithm(Coarser<GraphType, GraphType>& coarser)
{
    GraphType graph = SimpleGraphWithVertexTypes();

    GraphType coarseGraph;
    std::vector<VertexIdxT<GraphType>> contractionMap;

    EXPECT_TRUE(coarser.CoarsenDag(graph, coarseGraph, contractionMap));
    testValidContractionMap(graph, coarseGraph, contractionMap);
}

TEST_F(TestOspAlgorithms, CoarsenSarkar)
{
    sarkar_params::Parameters<VWorkwT<GraphType>> params;
    params.mode_ = sarkar_params::Mode::LINES;
    params.commCost_ = 100;
    params.useTopPoset_ = true;

    Sarkar<GraphType, GraphType> coarser(params);

    testCoarseningAlgorithm(coarser);

    params.useTopPoset_ = false;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::FAN_IN_FULL;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::FAN_IN_PARTIAL;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::FAN_OUT_FULL;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::FAN_OUT_PARTIAL;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::LEVEL_EVEN;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::LEVEL_ODD;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::FAN_IN_BUFFER;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::FAN_OUT_BUFFER;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.mode_ = sarkar_params::Mode::HOMOGENEOUS_BUFFER;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);
}

// Group seven structurally identical buffer nodes in every homogeneous-buffer mode.
TEST_F(TestOspAlgorithms, SarkarGroupsHomogeneousBuffers)
{
    constexpr VertType kBufferCount = 7;
    constexpr VertType kSink = kBufferCount + 1;
    std::set<std::pair<VertType, VertType>> edges;
    for (VertType buffer = 1; buffer <= kBufferCount; ++buffer) {
        edges.emplace(0, buffer);
        edges.emplace(buffer, kSink);
    }

    GraphType graph(kSink + 1, edges);
    for (const auto vertex : graph.Vertices()) {
        graph.SetVertexWorkWeight(vertex, 1);
    }

    sarkar_params::Parameters<VWorkwT<GraphType>> params;
    params.commCost_ = 1;
    params.maxWeight_ = 3;
    params.smallWeightThreshold_ = 3;
    const std::vector<sarkar_params::Mode> modes = {
        sarkar_params::Mode::FAN_IN_BUFFER,
        sarkar_params::Mode::FAN_OUT_BUFFER,
        sarkar_params::Mode::HOMOGENEOUS_BUFFER,
    };

    for (const auto mode : modes) {
        params.mode_ = mode;
        Sarkar<GraphType, GraphType> coarser(params);
        VertexIdxT<GraphType> contractions = 0;
        const auto expansionMap = coarser.GenerateVertexExpansionMap(graph, contractions);

        EXPECT_TRUE(coarser_util::CheckValidExpansionMap<GraphType>(expansionMap));
        EXPECT_GT(contractions, 0);
        EXPECT_TRUE(
            std::any_of(expansionMap.begin(), expansionMap.end(), [](const auto& group) { return group.size() > 1U; }));
    }
}

TEST_F(TestOspAlgorithms, SarkarContractsCompleteFanOut)
{
    const GraphType graph(4, std::set<std::pair<VertType, VertType>>({{0, 1}, {0, 2}, {0, 3}}));
    sarkar_params::Parameters<VWorkwT<GraphType>> params;
    params.mode_ = sarkar_params::Mode::FAN_OUT_FULL;
    params.commCost_ = 10;
    params.maxWeight_ = 4;

    Sarkar<GraphType, GraphType> coarser(params);
    VertexIdxT<GraphType> contractions = 0;
    const auto expansionMap = coarser.GenerateVertexExpansionMap(graph, contractions);

    ASSERT_EQ(expansionMap.size(), 1U);
    EXPECT_EQ(expansionMap.front().size(), graph.NumVertices());
    EXPECT_EQ(contractions, 3);
    EXPECT_TRUE(coarser_util::CheckValidExpansionMap<GraphType>(expansionMap));
}

TEST_F(TestOspAlgorithms, CoarsenSarkarML)
{
    sarkar_params::MulParameters<VWorkwT<GraphType>> params;
    params.commCostVec_ = {100};

    SarkarMul<GraphType, GraphType> coarser;
    coarser.SetParameters(params);
    testCoarseningAlgorithm(coarser);

    params.commCostVec_ = {1, 2, 10, 50, 100};
    params.bufferMergeMode_ = sarkar_params::BufferMergeMode::FULL;
    coarser.SetParameters(params);

    testCoarseningAlgorithm(coarser);
}

// Cover identity fallback, contraction-map composition, history compaction, and the prebuilt-graph overload.
TEST_F(TestOspAlgorithms, MultilevelCoarserMaintainsContractionHistory)
{
    const GraphType graph = SimpleGraph();
    const std::vector<TestMultilevelCoarser::Scenario> scenarios = {
        TestMultilevelCoarser::Scenario::NO_CONTRACTIONS,
        TestMultilevelCoarser::Scenario::COMPACT_IDENTITY,
        TestMultilevelCoarser::Scenario::KEEP_LARGE_REDUCTION,
        TestMultilevelCoarser::Scenario::PREBUILT_GRAPH,
    };

    for (const auto scenario : scenarios) {
        TestMultilevelCoarser coarser(scenario);
        GraphType coarseGraph;
        std::vector<VertexIdxT<GraphType>> contractionMap;

        EXPECT_TRUE(coarser.CoarsenDag(graph, coarseGraph, contractionMap));
        EXPECT_EQ(contractionMap.size(), graph.NumVertices());
        EXPECT_TRUE(coarser_util::CheckValidContractionMap<GraphType>(contractionMap));
        EXPECT_EQ(coarseGraph.NumVertices(), static_cast<VertexIdxT<GraphType>>(
                                                 *std::max_element(contractionMap.begin(), contractionMap.end()) + 1));

        if (scenario == TestMultilevelCoarser::Scenario::COMPACT_IDENTITY) {
            EXPECT_EQ(coarser.HistorySize(), 2U);
        } else if (scenario == TestMultilevelCoarser::Scenario::KEEP_LARGE_REDUCTION) {
            EXPECT_EQ(coarser.HistorySize(), 3U);
        } else {
            EXPECT_EQ(coarser.HistorySize(), 1U);
        }
    }
}

TEST_F(TestOspAlgorithms, DagAdaptorSimpleGraph)
{
    const std::vector<std::vector<VertType>> outEdges({{1, 2}, {2, 6}, {3}, {7}, {6}, {6}, {7, 10}, {9}, {}, {}, {}});

    const std::vector<std::vector<VertType>> inEdges({{}, {0}, {0, 1}, {2}, {}, {}, {1, 4, 5}, {3, 6}, {}, {7}, {6}});

    GraphAdapterType graph(outEdges, inEdges);

    std::size_t cntr0{};
    std::size_t cntrChldEdges{};
    std::size_t cntrParEdges{};
    for (const auto& vert : graph.Vertices()) {
        EXPECT_EQ(vert, cntr0++);
        cntrChldEdges += graph.OutDegree(vert);
        cntrParEdges += graph.InDegree(vert);
    }
    EXPECT_EQ(graph.NumEdges(), cntrChldEdges);
    EXPECT_EQ(graph.NumEdges(), cntrParEdges);

    for (const auto& vert : graph.Vertices()) {
        EXPECT_EQ(graph.InDegree(vert), inEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto& par : graph.Parents(vert)) {
            EXPECT_EQ(par, inEdges[vert][cntr++]);
        }
        EXPECT_EQ(cntr, graph.InDegree(vert));
    }

    for (const auto& vert : graph.Vertices()) {
        EXPECT_EQ(graph.OutDegree(vert), outEdges[vert].size());
        std::size_t cntr = 0;
        for (const auto& chld : graph.Children(vert)) {
            EXPECT_EQ(chld, outEdges[vert][cntr++]);
        }
        EXPECT_EQ(cntr, graph.OutDegree(vert));
    }

    for (const auto& vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexType(vert), 0);
    }

    for (const auto vert : graph.Vertices()) {
        graph.SetVertexWorkWeight(vert, 4 * vert + 0);
        graph.SetVertexCommWeight(vert, 4 * vert + 1);
        graph.SetVertexMemWeight(vert, 4 * vert + 2);
        graph.SetVertexType(vert, static_cast<VTypeType>(4 * vert + 3));
    }
    for (const auto vert : graph.Vertices()) {
        EXPECT_EQ(graph.VertexWorkWeight(vert), 4 * vert + 0);
        EXPECT_EQ(graph.VertexCommWeight(vert), 4 * vert + 1);
        EXPECT_EQ(graph.VertexMemWeight(vert), 4 * vert + 2);
        EXPECT_EQ(graph.VertexType(vert), static_cast<VTypeType>(4 * vert + 3));
    }
}

TEST_F(TestOspAlgorithms, MutableGraphRejectsInvalidAndDuplicateEdges)
{
    ConstrGraphType graph;
    graph.AddVertex(1, 1, 1);
    graph.AddVertex(1, 1, 1);

    EXPECT_FALSE(graph.AddEdge(0, 0));
    EXPECT_FALSE(graph.AddEdge(0, 2));
    EXPECT_TRUE(graph.AddEdge(0, 1));
    EXPECT_FALSE(graph.AddEdge(0, 1));
    EXPECT_EQ(graph.OutDegree(0), 1);
}

TEST_F(TestOspAlgorithms, BspSchedulers)
{
    BspInstance<GraphType> bspInst;
    bspInst.GetArchitecture() = BspArchitecture<GraphType>(3U);
    bspInst.GetComputationalDag() = SimpleGraph();

    BspSchedule<GraphType> schedule(bspInst);

    GrowLocalAutoCores<GraphType> growlocal;
    growlocal.ComputeSchedule(schedule);
    EXPECT_TRUE(schedule.IsValid());

    GreedyChildren<GraphType> children;
    children.ComputeSchedule(schedule);
    EXPECT_TRUE(schedule.IsValid());

    KlImprover<GraphType, KlHyperTotalCommCostFunction<GraphType, double, 1>, 1, double> kl;
    kl.SetSuperstepRemoveStrengthParameter(1.0);
    kl.SetTimeQualityParameter(1.0);

    ComboScheduler<GraphType> growlocalKl(growlocal, kl);
    growlocalKl.ComputeSchedule(schedule);
    EXPECT_TRUE(schedule.IsValid());

    ComboScheduler<GraphType> childrenKl(children, kl);
    childrenKl.ComputeSchedule(schedule);
    EXPECT_TRUE(schedule.IsValid());

    GreedyMetaScheduler<GraphType> greedymeta;
    greedymeta.AddScheduler(growlocalKl);
    greedymeta.AddScheduler(childrenKl);
    greedymeta.AddSerialScheduler();

    greedymeta.ComputeSchedule(schedule);
    EXPECT_TRUE(schedule.IsValid());
}

TEST_F(TestOspAlgorithms, SerialSchedulerDefersIncompatibleChildren)
{
    BspInstance<ConstrGraphType> instance;
    instance.GetArchitecture().SetProcessorsWithTypes({0U, 1U});
    auto& graph = instance.GetComputationalDag();
    graph.AddVertex(1, 1, 1, 0U);
    graph.AddVertex(1, 1, 1, 1U);
    ASSERT_TRUE(graph.AddEdge(0, 1));
    instance.SetDiagonalCompatibilityMatrix(2U);

    Serial<ConstrGraphType> scheduler;
    BspSchedule<ConstrGraphType> schedule(instance);
    EXPECT_EQ(scheduler.ComputeSchedule(schedule), ReturnStatus::OSP_SUCCESS);
    EXPECT_TRUE(schedule.IsValid());
    EXPECT_EQ(schedule.AssignedProcessor(0), 0U);
    EXPECT_EQ(schedule.AssignedProcessor(1), 1U);
    EXPECT_EQ(schedule.AssignedSuperstep(0), 0U);
    EXPECT_EQ(schedule.AssignedSuperstep(1), 1U);

    BspInstance<ConstrGraphType> emptyInstance;
    BspSchedule<ConstrGraphType> emptySchedule(emptyInstance);
    EXPECT_EQ(scheduler.ComputeSchedule(emptySchedule), ReturnStatus::OSP_SUCCESS);
}

TEST_F(TestOspAlgorithms, EftSchedulerReportsInsufficientWorkers)
{
    BspInstance<ConstrGraphType> instance;
    instance.GetArchitecture() = BspArchitecture<ConstrGraphType>(1U);
    instance.GetComputationalDag().AddVertex(4000, 1, 1);

    EftSubgraphScheduler<ConstrGraphType> scheduler;
    const auto schedule = scheduler.Run(instance, {2U}, {{4000}}, {2U});
    EXPECT_DOUBLE_EQ(schedule.makespan_, -1.0);
    EXPECT_TRUE(schedule.nodeAssignedWorkerPerType_.empty());
}

// Compare forward and backward Merkle hashing and reject graphs with different orbit structures.
TEST_F(TestOspAlgorithms, MerkleHashComputesForwardAndBackwardOrbits)
{
    const GraphType chain(4, std::set<std::pair<VertType, VertType>>({{0, 1}, {1, 2}, {2, 3}}));
    const GraphType fanOut(4, std::set<std::pair<VertType, VertType>>({{0, 1}, {0, 2}, {0, 3}}));
    const GraphType shortChain(3, std::set<std::pair<VertType, VertType>>({{0, 1}, {1, 2}}));

    EXPECT_TRUE(AreIsomorphicByMerkleHash(chain, chain));
    EXPECT_FALSE(AreIsomorphicByMerkleHash(chain, fanOut));
    EXPECT_FALSE(AreIsomorphicByMerkleHash(chain, shortChain));

    MerkleHashComputer<GraphType> forwardHashes(chain);
    EXPECT_EQ(forwardHashes.GetVertexHashes().size(), chain.NumVertices());
    EXPECT_EQ(forwardHashes.GetOrbit(0), forwardHashes.GetOrbitFromHash(forwardHashes.GetVertexHash(0)));
    EXPECT_EQ(forwardHashes.NumOrbits(), chain.NumVertices());
    EXPECT_THROW(forwardHashes.GetOrbitFromHash(std::numeric_limits<std::size_t>::max()), std::out_of_range);

    using UniformHash = UniformNodeHashFunc<VertexIdxT<GraphType>>;
    MerkleHashComputer<GraphType, UniformHash, false> backwardHashes(chain);
    BwdMerkleNodeHashFunc<GraphType> backwardNodeHash(chain);
    EXPECT_EQ(backwardHashes.GetVertexHash(0), backwardNodeHash(0));

    const std::vector<std::size_t> nodeHashes({2U, 3U, 5U, 7U});
    PrecomBwdMerkleNodeHashFunc<GraphType> precomputedHash(chain, nodeHashes);
    EXPECT_NE(precomputedHash(0), precomputedHash(3));
}

// Move an empty superstep in both directions and swap populated steps.
TEST_F(TestOspAlgorithms, ActiveScheduleMovesEmptyAndPopulatedSteps)
{
    BspInstance<GraphType> instance;
    instance.GetArchitecture() = BspArchitecture<GraphType>(2U);
    instance.GetComputationalDag() = GraphType(3, std::set<std::pair<VertType, VertType>>({}));
    const BspSchedule<GraphType> schedule(instance, {0U, 0U, 0U}, {1U, 2U, 3U});

    KlActiveSchedule<GraphType, double> activeSchedule;
    activeSchedule.Initialize(schedule);
    activeSchedule.SwapEmptyStepFwd(0U, 2U);
    EXPECT_EQ(activeSchedule.AssignedSuperstep(0), 0U);
    EXPECT_EQ(activeSchedule.AssignedSuperstep(1), 1U);

    activeSchedule.SwapEmptyStepBwd(2U, 0U);
    EXPECT_EQ(activeSchedule.AssignedSuperstep(0), 1U);
    EXPECT_EQ(activeSchedule.AssignedSuperstep(1), 2U);

    activeSchedule.SwapSteps(1U, 1U);
    activeSchedule.SwapSteps(1U, 3U);
    EXPECT_EQ(activeSchedule.AssignedSuperstep(0), 3U);
    EXPECT_EQ(activeSchedule.AssignedSuperstep(2), 1U);
}

TEST_F(TestOspAlgorithms, KlUtilitiesResizeAndSelectViolatingNodes)
{
    GraphType graph(3, std::set<std::pair<VertType, VertType>>({{0, 1}, {1, 2}}));
    BspInstance<GraphType> instance;
    instance.GetArchitecture() = BspArchitecture<GraphType>(2U);
    instance.GetComputationalDag() = graph;
    const BspSchedule<GraphType> schedule(instance, {0U, 1U, 0U}, {0U, 1U, 2U});

    KlActiveSchedule<GraphType, double> activeSchedule;
    activeSchedule.Initialize(schedule);
    using ActiveSchedule = KlActiveSchedule<GraphType, double>;
    using AffinityTable = AdaptiveAffinityTable<GraphType, double, ActiveSchedule, 1U>;
    AffinityTable affinityTable;
    affinityTable.Initialize(activeSchedule, 1U);

    EXPECT_TRUE(affinityTable.Insert(0));
    EXPECT_FALSE(affinityTable.Insert(0));
    EXPECT_TRUE(affinityTable.Insert(1));
    EXPECT_EQ(affinityTable.size(), 2U);
    affinityTable.Remove(0);
    EXPECT_TRUE(affinityTable.Insert(2));
    EXPECT_EQ(affinityTable.GetSelectedNodesIdx(2), 0U);
    affinityTable.Remove(2);
    affinityTable.Trim();
    EXPECT_EQ(affinityTable.GetSelectedNodesIdx(1), 0U);

    affinityTable.ResetNodeSelection();
    std::mt19937 generator(7U);
    VertexSelectionStrategy<GraphType, AffinityTable, ActiveSchedule> selectionStrategy;
    selectionStrategy.Initialize(activeSchedule, generator, 1U, 1U);
    std::unordered_set<EdgeDescT<GraphType>> violations({EdgeDescT<GraphType>(0, 1), EdgeDescT<GraphType>(1, 2)});
    selectionStrategy.SelectNodesViolations(affinityTable, violations, 1U, 1U);
    EXPECT_EQ(affinityTable.size(), 1U);
    EXPECT_TRUE(affinityTable.IsSelected(1));

    VectorVertexLockManager<VertexIdxT<GraphType>> lockManager;
    lockManager.Initialize(graph.NumVertices());
    lockManager.Lock(1);
    EXPECT_TRUE(lockManager.IsLocked(1));
    lockManager.Unlock(1);
    EXPECT_FALSE(lockManager.IsLocked(1));
    lockManager.Clear();
}

// Validate communication-cost bookkeeping against its independent recomputation path.
TEST_F(TestOspAlgorithms, KlCommunicationCostMatchesRecomputation)
{
    GraphType graph(3, std::set<std::pair<VertType, VertType>>({{0, 1}, {0, 2}}));
    graph.SetVertexCommWeight(0, 4);

    BspInstance<GraphType> instance;
    instance.GetArchitecture() = BspArchitecture<GraphType>(2U, 3, 5);
    instance.GetComputationalDag() = graph;
    const BspSchedule<GraphType> schedule(instance, {0U, 1U, 1U}, {0U, 1U, 1U});

    KlActiveSchedule<GraphType, double> activeSchedule;
    activeSchedule.Initialize(schedule);
    CompatibleProcessorRange<GraphType> processorRange(instance);
    KlHyperTotalCommCostFunction<GraphType, double, 1U> costFunction;
    costFunction.Initialize(activeSchedule, processorRange);

    const double cost = costFunction.ComputeScheduleCost();
    EXPECT_DOUBLE_EQ(cost, costFunction.ComputeScheduleCostTest());
    EXPECT_DOUBLE_EQ(costFunction.GetCommMultiplier(), 0.5);
    EXPECT_DOUBLE_EQ(costFunction.GetMaxCommWeight(), 4.0);
    EXPECT_DOUBLE_EQ(costFunction.GetMaxCommWeightMultiplied(), 2.0);
    EXPECT_EQ(costFunction.Name(), "hyper_total_comm_cost");

    using VertexType = VertexIdxT<GraphType>;
    std::map<VertexType, KlUpdateInfo<VertexType>> recompute;
    costFunction.MarkForFullRecompute(1, recompute);
    costFunction.MarkForFullRecompute(1, recompute);
    EXPECT_TRUE(recompute.at(1).fullUpdate_);

    const KlMoveStruct<double, VertexType> move(1, 0.0, 1U, 1U, 0U, 1U);
    static_cast<void>(costFunction.GetPreMoveCommData(move));
    costFunction.UpdateDatastructureAfterMove(move, 0U, 1U);
    costFunction.UpdateDatastructureAfterMove(move.ReverseMove(), 0U, 1U);
    EXPECT_DOUBLE_EQ(cost, costFunction.ComputeScheduleCostTest());
}

// Start from a valid but imbalanced schedule so KL can exercise real move and rollback decisions.
TEST_F(TestOspAlgorithms, KlImproverBalancesAValidSchedule)
{
    const std::set<std::pair<VertType, VertType>> edges({{0, 2}, {1, 2}, {2, 3}, {2, 4}, {3, 5}, {4, 5}});
    GraphType graph(6, edges);
    for (const auto vertex : graph.Vertices()) {
        graph.SetVertexWorkWeight(vertex, 10);
        graph.SetVertexCommWeight(vertex, 2);
    }

    BspInstance<GraphType> instance;
    instance.GetArchitecture() = BspArchitecture<GraphType>(3U, 1, 2);
    instance.GetComputationalDag() = graph;
    BspSchedule<GraphType> schedule(instance, {0U, 0U, 0U, 0U, 0U, 0U}, {0U, 0U, 1U, 2U, 2U, 3U});
    ASSERT_TRUE(schedule.IsValid());
    const auto initialCost = schedule.ComputeCosts();

    KlImprover<GraphType, KlHyperTotalCommCostFunction<GraphType, double, 1U>, 1U, double> improver(7U);
    improver.SetSuperstepRemoveStrengthParameter(1.0);
    improver.SetTimeQualityParameter(1.0);
    const ReturnStatus status = improver.ImproveSchedule(schedule);

    EXPECT_TRUE(status == ReturnStatus::OSP_SUCCESS || status == ReturnStatus::OSP_BEST_FOUND);
    EXPECT_TRUE(schedule.IsValid());
    EXPECT_LE(schedule.ComputeCosts(), initialCost);

    BspInstance<GraphType> singleProcessorInstance;
    singleProcessorInstance.GetArchitecture() = BspArchitecture<GraphType>(1U);
    singleProcessorInstance.GetComputationalDag() = graph;
    BspSchedule<GraphType> singleProcessorSchedule(singleProcessorInstance);
    EXPECT_EQ(improver.ImproveScheduleWithTimeLimit(singleProcessorSchedule), ReturnStatus::OSP_BEST_FOUND);
}

// Split two disconnected components over three processor groups and cover an unused group.
TEST_F(TestOspAlgorithms, TrimmedGroupSchedulerDistributesComponents)
{
    BspInstance<ConstrGraphType> instance;
    instance.GetArchitecture() = BspArchitecture<ConstrGraphType>(6U);
    auto& graph = instance.GetComputationalDag();
    for (unsigned i = 0; i < 4U; ++i) {
        graph.AddVertex(1, 1, 1);
    }
    EXPECT_TRUE(graph.AddEdge(0, 1));
    EXPECT_TRUE(graph.AddEdge(2, 3));

    BspSchedule<ConstrGraphType> schedule(instance);
    Serial<ConstrGraphType> serialScheduler;
    TrimmedGroupScheduler<ConstrGraphType> scheduler(serialScheduler, 3U);

    EXPECT_EQ(scheduler.ComputeSchedule(schedule), ReturnStatus::OSP_SUCCESS);
    EXPECT_TRUE(schedule.IsValid());
    EXPECT_EQ(schedule.NumberOfSupersteps(), 1U);
    EXPECT_EQ(schedule.AssignedProcessor(0), 0U);
    EXPECT_EQ(schedule.AssignedProcessor(1), 0U);
    EXPECT_EQ(schedule.AssignedProcessor(2), 2U);
    EXPECT_EQ(schedule.AssignedProcessor(3), 2U);
}

// An empty trimmed group is a successful zero-superstep schedule.
TEST_F(TestOspAlgorithms, TrimmedGroupSchedulerHandlesEmptyGraph)
{
    BspInstance<ConstrGraphType> instance;
    BspSchedule<ConstrGraphType> schedule(instance);
    Serial<ConstrGraphType> serialScheduler;
    TrimmedGroupScheduler<ConstrGraphType> scheduler(serialScheduler, 1U);

    EXPECT_EQ(scheduler.ComputeSchedule(schedule), ReturnStatus::OSP_SUCCESS);
    EXPECT_EQ(schedule.NumberOfSupersteps(), 0U);
}

TEST_F(TestOspAlgorithms, IsomorphicSchedulerTrimsRepeatedComponents)
{
    const std::set<std::pair<VertType, VertType>> edges({{0, 1}, {2, 3}, {4, 5}, {6, 7}});
    GraphType graph(8, edges);
    for (const auto vertex : graph.Vertices()) {
        graph.SetVertexWorkWeight(vertex, 2000);
    }
    BspInstance<GraphType> instance;
    instance.GetArchitecture() = BspArchitecture<GraphType>(4U);
    instance.GetComputationalDag() = graph;

    Serial<ConstrGraphType> serialScheduler;
    IsomorphicSubgraphScheduler<GraphType, ConstrGraphType> inferredHashScheduler(serialScheduler);
    inferredHashScheduler.SetWorkThreshold(100000);
    inferredHashScheduler.SetCriticalPathThreshold(100000);
    const auto inferredHashPartition = inferredHashScheduler.ComputePartition(instance);
    EXPECT_TRUE(coarser_util::CheckValidContractionMap<ConstrGraphType>(inferredHashPartition));

    MerkleHashComputer<GraphType> hashComputer(graph);
    IsomorphicSubgraphScheduler<GraphType, ConstrGraphType> trimmedScheduler(serialScheduler, hashComputer);
    trimmedScheduler.SetWorkThreshold(100000);
    trimmedScheduler.SetCriticalPathThreshold(100000);
    trimmedScheduler.EnableUseMaxGroupSize(2U);
    trimmedScheduler.SetAllowTrimmedScheduler(true);
    const auto trimmedPartition = trimmedScheduler.ComputePartition(instance);
    EXPECT_TRUE(coarser_util::CheckValidContractionMap<ConstrGraphType>(trimmedPartition));
}

TEST_F(TestOspAlgorithms, CoarsenMerkleBsp)
{
    BspInstance<GraphType> bspInst;
    bspInst.GetArchitecture() = BspArchitecture<GraphType>(3U);
    bspInst.GetArchitecture().SetProcessorsWithTypes(std::vector<unsigned>({0U, 0U, 1U}));
    bspInst.GetComputationalDag() = SimpleGraphWithVertexTypes();
    bspInst.SetDiagonalCompatibilityMatrix(2);

    GrowLocalAutoCores<ConstrGraphType> growlocal;
    GreedyChildren<ConstrGraphType> children;

    KlImprover<ConstrGraphType, KlHyperTotalCommCostFunction<ConstrGraphType, double, 1>, 1, double> kl;
    kl.SetSuperstepRemoveStrengthParameter(1.0);
    kl.SetTimeQualityParameter(1.0);

    ComboScheduler<ConstrGraphType> growlocalKl(growlocal, kl);
    ComboScheduler<ConstrGraphType> childrenKl(children, kl);

    GreedyMetaScheduler<ConstrGraphType> scheduler;
    scheduler.AddScheduler(growlocalKl);
    scheduler.AddScheduler(childrenKl);
    scheduler.AddSerialScheduler();

    std::vector<uint64_t> nodeHashList(bspInst.GetComputationalDag().NumVertices(), 7U);
    MerkleHashComputer<GraphType, PrecomBwdMerkleNodeHashFunc<GraphType>> hashComputer(
        bspInst.GetComputationalDag(), bspInst.GetComputationalDag(), nodeHashList);
    IsomorphicSubgraphScheduler<GraphType, ConstrGraphType> isoScheduler(scheduler, hashComputer);
    isoScheduler.SetWorkThreshold(200);
    isoScheduler.SetCriticalPathThreshold(500);

    const auto vertexContractionMap = isoScheduler.ComputePartition(bspInst);
    EXPECT_TRUE(coarser_util::CheckValidContractionMap<ConstrGraphType>(vertexContractionMap));

    ConstrGraphType coarseGraph;
    coarser_util::ConstructCoarseDag(bspInst.GetComputationalDag(), coarseGraph, vertexContractionMap);

    testValidContractionMap(bspInst.GetComputationalDag(), coarseGraph, vertexContractionMap);
}

} // namespace osp
} // namespace npu::tile_fwk
