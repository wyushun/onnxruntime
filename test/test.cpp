#include "cxxopts.hpp"

// clang-format off
void parse(int argc, const char* argv[], uint64_t &mb_size) {
  try {
    cxxopts::Options options(argv[0], " - npu command line options");
    options.positional_help("[optional args]").show_positional_help();

    options.set_width(70)
      .set_tab_expansion()
      .add_options()
      ("s,size", "Memory size(MB) to be operated", cxxopts::value<uint64_t>())
      ("h,help", "Print all command usages");

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      exit(0);
    }

    if (result.count("size")) {
      mb_size = result["size"].as<uint64_t>();
    } else {
      std::cout << options.help() << std::endl;
      exit(0);
    }

  } catch (const cxxopts::OptionException& e) {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(1);
  }
}
// clang-format on

int main(int argc, const char* argv[]) {
  uint64_t mb_size;
  parse(argc, argv, mb_size);

  return 0;
}

