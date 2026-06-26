#include "pead/arrow_io.hpp"

#include <memory>
#include <stdexcept>
#include <string>

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

namespace pead {
namespace {

[[noreturn]] void fail(const std::string& what, const arrow::Status& st) {
    throw std::runtime_error(what + ": " + st.ToString());
}

std::shared_ptr<arrow::ChunkedArray> column(const std::shared_ptr<arrow::Table>& t,
                                            const std::string& name) {
    auto col = t->GetColumnByName(name);
    if (!col) throw std::runtime_error("panel missing column: " + name);
    return col;
}

// Cast a column to `type` then flatten into a contiguous std::vector<CType>.
template <typename ArrayT, typename CType>
std::vector<CType> to_vector(const std::shared_ptr<arrow::ChunkedArray>& col,
                             const std::shared_ptr<arrow::DataType>& type) {
    arrow::Datum out;
    auto res = arrow::compute::Cast(col, type);
    if (!res.ok()) fail("cast", res.status());
    auto chunked = res.ValueOrDie().chunked_array();

    std::vector<CType> values;
    values.reserve(chunked->length());
    for (const auto& chunk : chunked->chunks()) {
        auto arr = std::static_pointer_cast<ArrayT>(chunk);
        for (int64_t i = 0; i < arr->length(); ++i) {
            values.push_back(arr->IsNull(i) ? CType(0) : static_cast<CType>(arr->Value(i)));
        }
    }
    return values;
}

}  // namespace

Panel read_panel(const std::string& path) {
    auto pool = arrow::default_memory_pool();

    auto infile_res = arrow::io::ReadableFile::Open(path, pool);
    if (!infile_res.ok()) fail("open " + path, infile_res.status());

    std::unique_ptr<parquet::arrow::FileReader> reader;
    auto st = parquet::arrow::OpenFile(infile_res.ValueOrDie(), pool, &reader);
    if (!st.ok()) fail("parquet open", st);

    std::shared_ptr<arrow::Table> table;
    st = reader->ReadTable(&table);
    if (!st.ok()) fail("read table", st);

    Panel p;
    p.secid    = to_vector<arrow::Int64Array, int64_t>(column(table, "secid"), arrow::int64());
    p.ann_date = to_vector<arrow::Date32Array, int32_t>(column(table, "ann_date"), arrow::date32());
    p.rel_day  = to_vector<arrow::Int32Array, int32_t>(column(table, "rel_day"), arrow::int32());
    p.impl_vol = to_vector<arrow::DoubleArray, double>(column(table, "impl_volatility"), arrow::float64());
    p.delta    = to_vector<arrow::DoubleArray, double>(column(table, "delta"), arrow::float64());
    p.volume   = to_vector<arrow::DoubleArray, double>(column(table, "volume"), arrow::float64());

    // cp_flag -> is_call (1 if "C"). Cast to string, then compare.
    auto cp = column(table, "cp_flag");
    auto cast = arrow::compute::Cast(cp, arrow::utf8());
    if (!cast.ok()) fail("cast cp_flag", cast.status());
    auto cp_str = cast.ValueOrDie().chunked_array();
    p.is_call.reserve(cp_str->length());
    for (const auto& chunk : cp_str->chunks()) {
        auto arr = std::static_pointer_cast<arrow::StringArray>(chunk);
        for (int64_t i = 0; i < arr->length(); ++i) {
            const auto v = arr->IsNull(i) ? std::string() : arr->GetString(i);
            p.is_call.push_back((!v.empty() && (v[0] == 'C' || v[0] == 'c')) ? 1 : 0);
        }
    }
    return p;
}

void write_results(const Results& r, const std::string& path) {
    arrow::Int64Builder secid_b;
    arrow::Date32Builder anndate_b;
    arrow::DoubleBuilder pre_b, post_b, drift_b, vol_b;
    arrow::Int64Builder npre_b, npost_b;

    for (std::size_t i = 0; i < r.size(); ++i) {
        (void)secid_b.Append(r.secid[i]);
        (void)anndate_b.Append(r.ann_date[i]);
        (void)pre_b.Append(r.atm_iv_pre[i]);
        (void)post_b.Append(r.atm_iv_post[i]);
        (void)drift_b.Append(r.iv_drift[i]);
        (void)npre_b.Append(r.n_pre[i]);
        (void)npost_b.Append(r.n_post[i]);
        (void)vol_b.Append(r.total_volume[i]);
    }

    std::shared_ptr<arrow::Array> secid, anndate, pre, post, drift, npre, npost, vol;
    (void)secid_b.Finish(&secid);
    (void)anndate_b.Finish(&anndate);
    (void)pre_b.Finish(&pre);
    (void)post_b.Finish(&post);
    (void)drift_b.Finish(&drift);
    (void)npre_b.Finish(&npre);
    (void)npost_b.Finish(&npost);
    (void)vol_b.Finish(&vol);

    auto schema = arrow::schema({
        arrow::field("secid", arrow::int64()),
        arrow::field("ann_date", arrow::date32()),
        arrow::field("atm_iv_pre", arrow::float64()),
        arrow::field("atm_iv_post", arrow::float64()),
        arrow::field("iv_drift", arrow::float64()),
        arrow::field("n_pre", arrow::int64()),
        arrow::field("n_post", arrow::int64()),
        arrow::field("total_volume", arrow::float64()),
    });
    auto table = arrow::Table::Make(
        schema, {secid, anndate, pre, post, drift, npre, npost, vol});

    auto outfile_res = arrow::io::FileOutputStream::Open(path);
    if (!outfile_res.ok()) fail("open out " + path, outfile_res.status());
    auto st = parquet::arrow::WriteTable(
        *table, arrow::default_memory_pool(), outfile_res.ValueOrDie(), 1 << 20);
    if (!st.ok()) fail("write table", st);
}

}  // namespace pead
