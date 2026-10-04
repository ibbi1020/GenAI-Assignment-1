import random

from genai.data.fs2k import count_styles, sketch_location, split_train_val as split_fs2k
from genai.data.pets import assert_disjoint, split_train_val as split_pets


def test_pet_split_is_twenty_percent_and_stable():
    ids = [f"pet_{index:03d}" for index in range(50)]
    train_a, val_a = split_pets(ids)
    train_b, val_b = split_pets(ids)

    assert train_a == train_b
    assert val_a == val_b
    assert len(val_a) == int(round(50 * 0.2))
    assert len(train_a) + len(val_a) == 50
    assert set(train_a).isdisjoint(set(val_a))
    assert train_a == sorted(train_a)
    assert val_a == sorted(val_a)
    assert_disjoint(train_a, val_a, ["held_out"])


def test_pet_split_follows_official_order_not_a_sorted_shuffle():
    ids = ["c", "a", "b"]
    shuffled = list(ids)
    random.Random(42).shuffle(shuffled)
    train_ids, val_ids = split_pets(ids, val_fraction=1 / 3)
    assert set(val_ids) == set(shuffled[:1])
    assert set(train_ids) == set(shuffled[1:])


def test_fs2k_validation_is_stratified_and_stable():
    records = []
    for style, count in ((0, 20), (1, 10), (2, 7)):
        for index in range(count):
            records.append({"image_name": f"photo{style}/img{index:03d}", "style": style})

    train_a, val_a = split_fs2k(records)
    train_b, val_b = split_fs2k(records)
    assert [row["image_name"] for row in train_a] == [row["image_name"] for row in train_b]
    assert [row["image_name"] for row in val_a] == [row["image_name"] for row in val_b]

    official_counts = count_styles(records)
    val_counts = count_styles(val_a)
    for style, count in official_counts.items():
        assert val_counts[style] == int(round(count * 0.15))

    train_names = {row["image_name"] for row in train_a}
    val_names = {row["image_name"] for row in val_a}
    assert train_names.isdisjoint(val_names)
    assert train_names | val_names == {row["image_name"] for row in records}


def test_fs2k_sketch_names_follow_the_photo_number():
    assert sketch_location("photo1/image0110") == ("sketch1", "sketch0110")
    assert sketch_location("photo2/image0007") == ("sketch2", "sketch0007")
    assert sketch_location("photo3/image0477") == ("sketch3", "sketch0477")
