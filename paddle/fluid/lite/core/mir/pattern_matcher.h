// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * This file implements PatternMatcher, which helps to operator on the patterns
 * in a DAG. It is similar to framework::ir::GraphPatternDetector, but due to
 * the size of the library, we need an simplified one, and more modular.
 *
 * This one is implemented specifically for Lite framework, and is possible to
 * execute on mobile, suitable both in storage and performance.
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/core/mir/node.h"
#include "paddle/fluid/lite/model_parser/compatible_pb.h"

namespace paddle {
namespace lite {
namespace mir {

/// The base class of all the Node implementation that can be processed with
/// PatternMatcher. The graph node such as the mir::Node should inherit it.
template <typename NodeT>
class NodePmBase {
 public:
  // Get the operator desc from an OpNode.
  const lite::OpDesc* op_desc() const { return self()->op_desc(); }
  // Get the variable desc from an VarNode.
  const lite::VarDesc* var_desc() const { return self()->var_desc(); }

 private:
  NodeT* self() { return static_cast<NodeT*>(this); }
};

/// The base class for all the Node for defining a PatternMatcher pattern.
template <typename NodeT>
class PmNodeBase {
 public:
 private:
  NodeT* self() { return static_cast<NodeT*>(this); }
};

template <typename PmNodeT>
class PmPattern;
/// The node representation for PatternMatcher. User can set assertion on the
/// OpDesc or VarDesc, or the KernelLite information.
template <typename NodeT>
class PmNode : public NodeBase<PmNode<NodeT>> {
  // tell whether an ir::Node* is a candidation for a PmNode.
  using teller_t = std::function<bool(NodeT*)>;
  using node_t = NodeT;
  using self_t = PmNode<node_t>;
  using pm_pattern_t = PmPattern<PmNode<NodeT>>;

  enum class Type { kOp, kVar };
  enum class Role {
    kUnknown,      // No role,
    kInput,        // an input and will be retained,
    kOutput,       // an output and will be retained,
    kIntermediate  // will be removed after handler.
  };

  void LinkTo(self_t* x);
  void LinkFrom(self_t* x);

  // this link to others
  PmNode& LinksTo(const std::vector<PmNode*>& others);

  PmNode& LinksFrom(const std::vector<PmNode*>& others);

  bool Tell(node_t* node) const {
    if (teller_) return teller_(node);

    for (auto& asrt : asserts_) {
      if (!asrt(node)) return false;
    }
    return true;
  }

  bool IsOp() const { return type_ == Type::kOp; }
  bool IsVar() const { return type_ == Type::kVar; }

  const std::string& name() const { return name_; }

  PmNode& operator=(const PmNode&) = delete;
  PmNode(const PmNode&) = delete;

  // Mark this node is an Input of a subgraph and will be retained.
  node_t* AsInput() {
    role_ = Role::kInput;
    return this;
  }
  // Mark this node is an Output of a subgraph and will be retained.
  node_t* AsOutput() {
    role_ = Role::kOutput;
    return this;
  }
  // Mark this node will be removed, so all the links should be inside a matched
  // sub-graph.
  PmNode* AsIntermediate() {
    role_ = Role::kIntermediate;
    return this;
  }

  bool IsIntermediate() const { return role_ == Role::kIntermediate; }
  bool IsInput() const { return role_ == Role::kInput; }
  bool IsOutput() const { return role_ == Role::kOutput; }

 private:
  PmNode(pm_pattern_t* pattern, const std::string& name = "",
         Type type = Type::kVar)
      : pattern_(pattern), name_(name), type_(type) {}
  PmNode(teller_t&& teller, pm_pattern_t* pattern, const std::string& name = "",
         Type type = Type::kVar)
      : teller_(std::move(teller)),
        pattern_(pattern),
        name_(name),
        type_(type) {
    PADDLE_ENFORCE(teller_ != nullptr, "invalid teller functer is set.");
  }

  PmNode(PmNode&& other) = default;

  friend class PDPattern;

  // Will removed latter.
  teller_t teller_;
  std::vector<teller_t> asserts_;
  pm_pattern_t* pattern_{};
  std::string name_;
  Type type_;
  Role role_{Role::kUnknown};
};

/// A pattern representation for PatternMatcher.
template <typename PmNodeT>
class PmPattern {
 public:
  using edge_t = std::pair<PmNodeT*, PmNodeT*>;
  using pm_node_t = PmNodeT;

  void AddEdge(PmNodeT* a, PmNodeT* b) { edges_.emplace_back(a, b); }

  PmNodeT* NewNode(typename PmNodeT::teller_t&& teller,
                   const std::string& name = NewID()) {
    if (!name.empty()) {
      PADDLE_ENFORCE_EQ(node_map_.count(name), 0UL,
                        "PDNode's name should be unique, get duplicate [%s]",
                        name);
    }

    nodes_.emplace_back(new pm_node_t(std::move(teller), this, name));
    auto* cur = nodes_.back().get();
    node_map_[name] = cur;
    return cur;
  }

  PmNodeT* NewNode(const std::string& name = NewID()) {
    if (!name.empty()) {
      PADDLE_ENFORCE_EQ(node_map_.count(name), 0UL,
                        "PDNode's name should be unique, get duplicate [%s]",
                        name);
    }

    nodes_.emplace_back(new pm_node_t(this, name));
    auto* cur = nodes_.back().get();
    node_map_[name] = cur;
    return cur;
  }
  PmNodeT* NewNode(const std::string& prefix, const std::string& name) {
    return NewNode(prefix + "/" + name);
  }
  PmNodeT* RetrieveNode(const std::string& id) const {
    auto it = node_map_.find(id);
    if (it == node_map_.end()) return nullptr;
    return *it;
  }

  const std::vector<std::unique_ptr<PmNodeT>>& nodes() const { return nodes_; }
  const std::vector<edge_t>& edges() const { return edges_; }

  std::string DotString() const;

 private:
  static std::string NewID() { return "pmnode-" + std::to_string(id_++); }

  std::vector<std::unique_ptr<PmNodeT>> nodes_;
  std::vector<edge_t> edges_;
  std::unordered_map<std::string, PmNodeT*> node_map_;
  static size_t id_;
};

/*
 * PatternMatcher is used to automatically extract some patterns from a graph.
 * The pattern is PmPattern, which is pre-defined using PmNode and edges.
 *
 * @PmNodeT: the type of the pattern matcher node.
 * @GraphT: the type of a graph.
 * @NodeT: the node of a graph.
 */
template <typename PmNodeT, typename GraphT, typename NodeT>
class PatternMatcher {
 public:
  using pm_node_t = PmNodeT;
  using pm_pattern_t = PmPattern<pm_node_t>;
  using node_t = NodeT;
  using graph_t = GraphT;
  using subgraph_t = std::unordered_map<PmNodeT*, NodeT*>;

  // Operate on the detected pattern.
  using handle_t =
      std::function<void(const subgraph_t& /*hitted pattern*/, graph_t*)>;

  void operator()(GraphT* graph, handle_t handler);

  const pm_pattern_t& pattern() const { return pattern_; }
  pm_pattern_t* mutable_pattern() { return &pattern_; }

 private:
  // Mark the nodes that fits the pattern.
  bool MarkPDNodesInGraph(const graph_t& graph);

  // Detect all the pattern and output the hit records.
  std::vector<subgraph_t> DetectPatterns();

  // Remove duplicate patterns.
  void UniquePatterns(std::vector<subgraph_t>* subgraphs);

  // Remove overlapped match subgraphs, when overlapped, keep the previous one.
  // The intermediate PmNodeT will be removed, so can't shared by multiple
  // patterns.
  void RemoveOverlappedMatch(std::vector<subgraph_t>* subgraphs);

  // Validate whether the intermediate nodes are linked by external nodes.
  void ValidateByNodeRole(std::vector<subgraph_t>* subgraphs);

 private:
  using hit_rcd_t =
      std::pair<node_t* /*node in graph*/, PmNodeT* /*node in pattern*/>;
  pm_pattern_t pattern_;
  std::unordered_map<const PmNodeT*, std::unordered_set<node_t*>>
      pdnodes2nodes_;
};

template <typename NodeT>
void PmNode<NodeT>::LinkTo(PmNode::self_t* x) {
  pattern_->AddEdge(this, x);
}
template <typename NodeT>
void PmNode<NodeT>::LinkFrom(PmNode::self_t* x) {
  pattern_->AddEdge(x, this);
}
template <typename NodeT>
PmNode<NodeT>& PmNode<NodeT>::LinksTo(const std::vector<PmNode*>& others) {
  for (auto* x : others) {
    LinkTo(x);
  }
}
template <typename NodeT>
PmNode<NodeT>& PmNode<NodeT>::LinksFrom(const std::vector<PmNode*>& others) {
  for (auto* x : others) {
    LinksFrom(x);
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
