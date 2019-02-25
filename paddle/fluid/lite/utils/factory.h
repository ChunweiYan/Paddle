#pragma once
#include <memory>
#include <unordered_map>

namespace paddle {
namespace lite {

template <typename ItemType>
class Factory {
 public:
  using item_t = ItemType;
  using self_t = Factory<item_t>;
  using item_ptr_t = std::unique_ptr<item_t>;
  using creator_t = std::function<item_ptr_t()>;

  static Factory& Global() {
    static Factory* x = new self_t;
    return *x;
  }

  void Register(const std::string& op_type, creator_t&& creator) {
    CHECK(!creators_.count(op_type)) << "The op " << op_type
                                     << " has already registered";
    creators_.emplace(op_type, std::move(creator));
  }

  item_ptr_t Create(const std::string& op_type) const {
    auto it = creators_.find(op_type);
    CHECK(it != creators_.end());
    return it->second();
  }

 protected:
  std::unordered_map<std::string, creator_t> creators_;
};

/* A helper function to help run a lambda at the start.
 */
template <typename Type>
class Registor {
 public:
  Registor(std::function<void()>&& functor) { functor(); }

  int Touch() { return 0; }
};

}  // namespace lite
}  // namespace paddle
